from subprocess import Popen, PIPE
import pandas as pd
import os,sys
import random
import pickle
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
from src.utils.get_action_space import *
import copy
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from scipy import optimize
#import threading
#import queue
from multiprocessing import Process, Queue
'''
def softmax(a) :
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
'''
use_maestro = False

action_space, action_bound, action_bottom = get_action_space()
action_space = [act * bound for act, bound in zip(action_space, action_bound)]
start_range, end_range = 0, len(action_space[0])

#from TimeloopMapping import *

from HWGene import _HWGene, HW_GENE
HWGene = _HWGene()
from MapGene import _MapGene, MAPPING_GENE
MapGene = _MapGene()

from TimeloopEstimation import TimeloopEstimator
from SparseloopEstimation import SparseloopEstimator
from FPGAConstraint.AlveoU200 import Constraint_AlveoU200_Sparse

class MaestroEnvironment(object):


    def __init__(self, model_defs, finish_reward=100, dim_size=6,n_action_steps=2, resource_size=2, dataflow="dla", is_discrete=True):
        super(MaestroEnvironment,self).__init__()
        dst_path = "../../cost_model/maestro"

        maestro = dst_path
        self._executable = "{}".format(maestro)
        random.seed()
        random_file_name = random.randint(0, 2 ** 31)
        self.random_file_name = "{}".format(random_file_name)

        self.fpga_constraint = Constraint_AlveoU200_Sparse()
        #self.timeloop_estimator = TimeloopEstimator(self.random_file_name)
        self.timeloop_estimator = SparseloopEstimator(self.random_file_name) #Sparse-aware

        self.is_gemm = False

        self.state = np.array([0.5]*8)
        self.last_runtime = 2 ** 64
        self.last_energy = 2**64
        self.last_throughput = 1
        self.observation = [0,0, 0,0,0,0]
        self.resource_state = [0, 0]
        self.consecutive_fail = 0
        self.max_fail = 0
        self.total_resources = action_bound
        self.action_bound = action_bound[:n_action_steps]
        self.total_step = len(model_defs)
        self.model_defs = model_defs
        self.model_defs_saved = copy.deepcopy(model_defs)
        model_bound = np.max(model_defs, axis=0, keepdims=True)
        self.model_defs_norm = model_defs/model_bound
        self.model_defs_norm_saved = copy.deepcopy(self.model_defs_norm)
        self.best_reward_whole = float("-inf")
        self.best_rewards = []
        self.best_rewards_constraint = []
        self.best_sol = None
        self.reward = 0
        self.mae_reward = 0
        self.finish_reward = finish_reward
        self.mae_reward_decay = 0.
        self.worst_reward = None
        self.best_reward = None
        self.reward_scale = 0
        self.n_action_steps = n_action_steps
        self.emptyinfo =  [-1] * (len(self.observation) + len(self.resource_state))
        self.resource_size = resource_size
        self.reward_whole_eps = 0
        self.dim_size = dim_size
        self.reward_rec = []
        self.min_reward = None
        self.running_ave_reward = None
        self.worst_reward_list = [None for _ in range(self.total_step)]
        self.sig = 1
        self.mac_rec = []
        self.count_invalid=0
        self.best_rewards_iteration = []
        self.epoch = 0
        self.sol_record = []
        self.exp_table = {}
        self.sol_reward_record = []
        self.dataflow = dataflow
        self.constraint_value = 2**63
        self.constraint = "area"
        self.prev_reward_whole_eps = 0
        self.exp_table = {}
        self.draw = np.arange(0,self.total_step )
        self.is_discrete = is_discrete
        self.state_size = len(self.model_defs_norm[0]) + 3
        self.update_best_sol = False

        self.map_best_reward = [None for _ in range(len(self.model_defs))]
        self.map_best_sol = [None for _ in range(len(self.model_defs))]
        self.map_best_rewards_iteration = [[] for _ in range(len(self.model_defs))]
        self.map_best_rewards = [[] for _ in range(len(self.model_defs))]
        self.best_reward_constraint = 0


    def reset(self):

        self.update_best_sol = False
        self.mac_rec = []
        self.sig = 1
        self.reward_record = []
        self.reward = 0
        self.sol = []
        self.mode = 0
        self.actions_step = 0
        action_idx = [3, 3]
        self.action = np.array([action_space[idx][val] for idx, val in enumerate(action_idx)])
        self.left_resource = [1 for _ in range(self.resource_size)]
        dimensions = self.model_defs_norm[self.mode]
        self.state = np.zeros((self.state_size,), dtype=np.float32)
        self.total_eps_rewards = 0
        return self.state

    def get_ref_constraint(self, bound=action_bound):
        raise 'Depl'
        #sol = [bound[:self.n_action_steps] for i in range(len(self.model_defs))]
        #_, total_constraint = self.exterior_search(sol)
        return total_constraint
    def set_constraint_value(self, max_constraint, min_constraint, constraint_value):
        self.constraint_value = constraint_value
        self.constraint_info = {"value": constraint_value,
                                "max": max_constraint,
                                "min": min_constraint}
    def set_constraint(self, constraint="area"):
        self.constraint = constraint
    def set_fitness(self, fitness="energy"):
        self.fitness = fitness




    def resource_check(self):
        return not any(np.array(self.left_resource) < 0)

    def judge(self, observation):
        #runtime, throughput, energy, area, l1_size, l2_size, mac, power = self.observation
        runtime, throughput, energy, area, l1_size, l2_size, mac, power = observation
        values = []
        for term in [self.fitness, self.constraint]:
            if term == "energy":
                reward = -energy
            elif term == "thrpt_ave":
                reward = throughput
            elif term == "LEP":
                reward = -energy * runtime
            elif term == "LAP":
                reward = -area * runtime
            elif term == "EAP":
                reward = -area * energy
            elif term == "thrpt" or term == "thrpt_naive":
                reward = throughput
            elif term == "thrpt_btnk":
                reward = throughput
            elif term == "latency":
                reward = -runtime
            elif term == "area":
                reward = -area
            elif term == "l1_size":
                reward = - l1_size
            elif term == "l2_size":
                reward = -l2_size
            elif term == "power":
                reward = -power
            else:
                raise NameError('Undefined fitness type')
            values.append(reward)
        return values[0], abs(values[1])


    def check_constraint(self, actions):
        used = np.sum(actions, axis=0)
        if any(used > self.resource_bound):
            return False
        return True

    def init_population(self, num_pop, num_layers):
        # Allocate&Initialize VALID population
        hw_new_population = np.empty((num_pop,len(HW_GENE)),dtype=float) # allocation
        map_new_population = np.empty((num_layers, num_pop,len(MAPPING_GENE)),dtype=float) # allocation
        sample_hw_gene = np.array(HWGene.get_sample_gene(), dtype=float).copy()
        sample_map_gene = np.array(MapGene.get_sample_gene(), dtype=float).copy()

        thread_number = 16
        count = 0
        pbar = tqdm(total=num_pop, desc="Initalize New HW Populations")
        #for iter_n in tqdm(range(num_pop), desc="Initalize New HW Populations"):
        while count<num_pop:
            # PE Pop
            reward = None
            constraint = None
            #que = queue.Queue()
            que = Queue()
            threads = []
            rand_gene = []
            for n_ in range(thread_number):
                rand_gene.append(HWGene.generate_random_gene())
                #t = threading.Thread(target=self.exterior_search, args=(self.model_defs[0], rand_gene[n_], sample_map_gene, n_, que))
                t = Process(target=self.exterior_search, args=(0, self.model_defs[0], rand_gene[n_], sample_map_gene, n_, que))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            for n_ in range(thread_number):
                i_, reward, constraint = que.get()
                if not(reward==None and (constraint==None or constraint > self.constraint_value)) and count<num_pop: #select valid gene
                    hw_new_population[count] = np.array(rand_gene[i_], dtype=float).copy()
                    count += 1
                    pbar.update(1)
        pbar.close()

        thread_number = 256#16*8
        count = 0
        pbar = tqdm(total=num_layers*num_pop, desc="Initalize New Map Populations")
        while count<num_layers*num_pop:
            # Map Pop for each layers
            reward = None
            #que = queue.Queue()
            que = Queue()
            threads = []
            rand_gene = []
            for n_ in range(thread_number):
                rand_gene.append(MapGene.generate_random_gene())
                #t = threading.Thread(target=self.exterior_search, args=(self.model_defs[0], sample_hw_gene, rand_gene[n_], n_, que))
                #t = Process(target=self.exterior_search, args=(self.model_defs[0], sample_hw_gene, rand_gene[n_], n_, que))
                ly_n = random.randint(0, num_layers-1)
                h_g_n = random.randint(0, num_pop-1)
                t = Process(target=self.exterior_search, args=(0, self.model_defs[ly_n], hw_new_population[h_g_n], rand_gene[n_], n_, que))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            for n_ in range(thread_number):
                i_, reward, constraint = que.get()
                if not(reward==None and (constraint==None or constraint > self.constraint_value)) and count<num_layers*num_pop: #select valid gene
                    map_new_population[count//num_pop][count%num_pop] = np.array(rand_gene[i_], dtype=float).copy()
                    count += 1
                    pbar.update(1)
        pbar.close()

        '''
        for count in tqdm(range(num_pop), desc="Initalize New Map Populations"):
            # Map Pop for each layers
            for i in range(num_layers):
                reward = None
                while reward==None: #select valid gene
                    rand_gene = MapGene.generate_random_gene()
                    map_new_population[i][count] = np.array(rand_gene, dtype=float).copy()
                    if use_maestro is True:
                        reward, constraint = self.oberserve_maestro(self.model_defs[0], sample_hw_gene, map_new_population[i][count])
                    else:
                        reward, constraint = self.exterior_search(self.model_defs[0], sample_hw_gene, map_new_population[i][count])
            count += 1
        '''
        return hw_new_population, map_new_population

    def get_fitness(self, gen, num_pop, num_layers, hw_new_population, map_new_population):
        # get HW/Mapping fitness
        hw_fitness = np.empty(num_pop, float)
        map_fitness = np.empty((num_layers, num_pop), float)
        invalid_count = 0
        #run
        #que = queue.Queue()
        pbar = tqdm(total=num_layers*num_pop, desc="GA: Get fitness")
        max_num_thread = 256
        n_ = 0
        reward = np.empty((num_pop, num_layers), float)
        constraint = np.empty((num_pop, num_layers), float)
        while True:
            que = Queue()
            threads = []
            for th_ in range(max_num_thread):
                i = n_//num_layers
                j = n_%num_layers
                hw_gene = hw_new_population[i]
                map_gene = map_new_population[j][i]
                #t = threading.Thread(target=self.exterior_search, args=(self.model_defs[j], hw_gene, map_gene, (i*(num_layers)+j), que))
                t = Process(target=self.exterior_search, args=(gen, self.model_defs[j], hw_gene, map_gene, (i*(num_layers)+j), que))
                t.start()
                threads.append(t)
                n_ += 1
                if n_ >= num_layers*num_pop: break
            ln = len(threads)
            for t in threads:
                t.join()
                pbar.update(1)
            for thn_ in range(ln):
                result_n_, reward_, constraint_ = que.get()
                if reward_  == None or constraint_ > self.constraint_value:
                    reward_ = float("-inf")
                    #constraint_ = float("-inf")
                i = result_n_//num_layers
                j = result_n_%num_layers
                reward[i][j] = reward_
                constraint[i][j] = constraint_
            if n_ >= num_layers*num_pop: break
        pbar.close()
        #get fitness
        '''
        for i_ in range(num_pop):
            for j_ in range(num_layers):
                n_, reward_, constraint_ = que.get()
                if reward_  == None or constraint_ > self.constraint_value:
                    reward_ = float("-inf")
                    #constraint_ = float("-inf")
                i = n_//num_layers
                j = n_%num_layers
                reward[i][j] = reward_
                constraint[i][j] = constraint_
        '''
        for i in range(num_pop):
            tot_reward = 0
            tot_constraint = 0
            for j in range(num_layers):
                map_fitness[j][i] = reward[i][j]
                #print(map_fitness[j][i])
                if reward[i][j] is float("-inf") or -(10**20)>reward[i][j] or tot_reward is float("-inf"):
                    tot_reward = float("-inf")
                else:
                    tot_reward += reward[i][j]
                tot_constraint = constraint[i][j]
            if tot_reward is None or tot_reward is float("-inf") or -(10**20)>tot_reward or tot_reward is float("inf"): # Can't compilation
                tot_reward = float("-inf")
                invalid_count += 1
            elif tot_constraint is None or tot_constraint > self.constraint_value:
                tot_reward = float("-inf")
                invalid_count += 1
            hw_fitness[i] = tot_reward
        print("Invalid rate: {:.2f}%({}/{})".format(invalid_count/num_pop*100, invalid_count, num_pop))

        '''
        for i in tqdm(range(num_pop), desc="initial fitness"):
            hw_gene = hw_new_population[i]

            tot_reward = 0
            tot_constraint = 0
            for j in range(num_layers):
                reward, constraint = self.exterior_search(self.model_defs[j], hw_gene, map_new_population[j][i])
                if reward == None or constraint > self.constraint_value:
                    reward = float("-inf")
                    #constraint = float("-inf")
                map_fitness[j][i] = reward
                tot_reward += reward
                tot_constraint = constraint
            reward = tot_reward
            total_used_constraint = tot_constraint
            if reward is None or reward is float("-inf"): # Can't compilation
                reward = float("-inf")
                #print("Error with reward")
                #print(new_population[i])
                #exit(-1)
                invalid_count += 1
            elif tot_constraint is None or total_used_constraint > self.constraint_value:
                reward = float("-inf")
                invalid_count += 1
            hw_fitness[i] = reward
        print("Invalid rate: {:.2f}%({}/{})".format(invalid_count/num_pop*100, invalid_count, num_pop))
        '''
        return hw_fitness, map_fitness

    def genetic_search(self, epochs=100, chkpt_file="genetic_chkpt.plt",fd=None):
        import logging
        import time
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logging_file = "logs/output_{}.log".format(time.strftime('%Y%m%d_%H%M%S'))
        os.remove(logging_file)  if os.path.exists(logging_file) else None
        output_file_handler = logging.FileHandler(logging_file)
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(output_file_handler)
        logger.addHandler(stdout_handler)

        epochs = 7000#3600 #hardcoded
        num_pop = 100#60 #hardcoded
        num_gen = epochs // num_pop
        num_parents = 40#30 #hardcoded
        self.fd = fd
        self.chkpt_file = chkpt_file
        self.start_range = start_range
        self.end_range = end_range-1

        if self.best_reward is None:
            self.best_reward = float("-inf")
        self.best_rewards = []
        for i in range(len(self.model_defs)):
            if self.map_best_reward[i] is None:
                self.map_best_reward[i] = float("-inf")
        self.epoch = 0
        self.best_sol = (HWGene.get_sample_gene(), [MapGene.get_sample_gene()])
        num_layers = len(self.model_defs)

        self.num_generations = num_gen
        self.num_population = num_pop
        self.num_parents = num_parents


        logger.debug("Number of generations: " + str(num_gen))
        logger.debug("Number of population: " + str(num_pop))
        logger.debug("Number of parents: " + str(num_parents))

        # Allocate population & init
        hw_new_population, map_new_population = self.init_population(num_pop, num_layers)
        logger.debug("[SYSTEM] Generated intial {} population".format(num_pop))
        

        # Get HW&mapping fitness for new pop
        #hw_fitness = self.get_HW_fitness(num_pop, num_layers, hw_new_population, map_new_population)
        #logger.debug("\n\n\n")
        #map_fitness = self.get_Mapping_fitness(num_pop, num_layers, hw_new_population, map_new_population)
        hw_fitness, map_fitness = self.get_fitness(0, num_pop, num_layers, hw_new_population, map_new_population)
        logger.debug("\n\n\n\n")

        # HW -> Mapping Opt.
        iteration = 0

        for generation in range(num_gen):
            print("Gen ", generation)
            if generation == 10:
                    self.best_reward = float("-inf")
            best_gen_reward =None
            parents = HWGene.select_parents(hw_new_population, hw_fitness, num_parents)

            offspring_crossover = HWGene.crossover(parents,
                                            num_pop-num_parents)
            offspring_mutation = HWGene.mutation(offspring_crossover, rate=0.1)

            hw_new_population[0:parents.shape[0], :] = parents
            hw_new_population[parents.shape[0]:, :] = offspring_mutation

            for lr_i in range(num_layers):
                parents = MapGene.select_parents(map_new_population[lr_i], map_fitness[lr_i],
                                                        num_parents)
                offspring_crossover = MapGene.crossover(parents,
                                                        num_pop-num_parents)
                offspring_mutation = MapGene.mutation(offspring_crossover, rate=0.1)

                map_new_population[lr_i][0:parents.shape[0], :] = parents
                map_new_population[lr_i][parents.shape[0]:, :] = offspring_mutation

            hw_fitness, map_fitness = self.get_fitness(generation, num_pop, num_layers, hw_new_population, map_new_population)
            for pop in range(num_pop):
                reward = hw_fitness[pop]
                if reward > self.best_reward:
                    best_gen_reward = reward
                    self.best_reward = reward
                    self.best_reward_constraint = float("inf") #total_used_constraint/num_layers
                    self.best_sol = (hw_new_population[pop].copy(), [map_new_population[lr_i][pop].copy() for lr_i in range(num_layers)])
                iteration += 1
                self.best_rewards_iteration.append(iteration)
                self.best_rewards_constraint.append(self.best_reward_constraint)
                self.best_rewards.append(self.best_reward)

            '''
            num_invalid_node = 0
            for pop in tqdm(range(num_pop), desc='GA population'):
                tot_hw_reward = 0.0
                total_used_constraint = 0.0

                hw_gene = hw_new_population[pop] #Select HW Gene
                
                if False:
                    for lr_i in range(num_layers):
                        map_gene = map_new_population[lr_i][pop]
                        reward, used_constraint = self.exterior_search(self.model_defs[lr_i], hw_gene, map_gene)
                        if reward is None or used_constraint is None: # invalid mapping
                            reward = float("-inf")
                            used_constraint = float("-inf")
                        elif used_constraint > self.constraint_value: # not met constraints
                            reward = float("-inf")
                            used_constraint = float("-inf")
                        map_fitness[lr_i][pop] = reward
                        tot_hw_reward += reward
                        total_used_constraint += used_constraint
                else:
                    que = queue.Queue()
                    threads = []
                    for lr_i in range(num_layers):
                        map_gene = map_new_population[lr_i][pop]
                        t = threading.Thread(target=self.exterior_search, args=(self.model_defs[lr_i], hw_gene, map_gene, lr_i, que))
                        t.start()
                        threads.append(t)
                    for t in threads:
                        t.join()
                    #for lr_i in range(num_layers):
                    for i_ in range(num_layers):
                        lr_i, reward, used_constraint = que.get()
                        #print(lr_i, reward, used_constraint)
                        if reward is None or used_constraint is None: # invalid mapping
                            reward = float("-inf")
                            used_constraint = float("-inf")
                        elif used_constraint > self.constraint_value: # not met constraints
                            reward = float("-inf")
                            used_constraint = float("-inf")
                        map_fitness[lr_i][pop] = reward
                        tot_hw_reward += reward
                        total_used_constraint += used_constraint

                reward = tot_hw_reward
                if reward is None or reward is float("-inf"): # invalid mapping
                    reward = float("-inf")
                    num_invalid_node += 1
                elif total_used_constraint//num_layers > self.constraint_value: # not met constraints
                    reward = float("-inf")
                    num_invalid_node += 1
                if reward > self.best_reward:
                    best_gen_reward = reward
                    self.best_reward = reward
                    self.best_reward_constraint = total_used_constraint/num_layers
                    self.best_sol = (hw_new_population[pop].copy(), [map_new_population[lr_i][pop].copy() for lr_i in range(num_layers)])

                hw_fitness[pop] = reward
                iteration += 1
                self.best_rewards_iteration.append(iteration)
                self.best_rewards_constraint.append(self.best_reward_constraint)
                self.best_rewards.append(self.best_reward)
            '''
            best_tmp_results = self.timeloop_estimator.get_gene_HW_info(self.model_defs[0], self.best_sol[0], self.best_sol[1][0])
            print("Best sol(Xdim, X, Ydim, Y, group_density, bank): ", best_tmp_results)
            if best_gen_reward  is not None:
                logger.debug("\nHW Generation {}: new best award reward: {:9e}".format(generation+1, self.best_reward))
            #self.count_invalid += num_invalid_node
            self.save_chkpt()
        # == end of HW GA

        self.save_chkpt()
        logger.debug("Best result total:")
        logger.debug(self.best_rewards)
        logger.debug("Constraints of best")
        logger.debug(self.best_rewards_constraint)

        import shutil
        os.mkdir('../../data/best') if os.path.exists("../../data/best") is False else None
        total_reward = 0
        print("best reward each layer")
        for lr_i in range(num_layers):
            reward, constraint = self.exterior_search(num_gen-1, self.model_defs[lr_i], self.best_sol[0], self.best_sol[1][lr_i])
            print(reward)
            total_reward += reward
            dir_name = '../../data/best/{:02d}'.format(lr_i)
            os.mkdir(dir_name) if os.path.exists(dir_name) is False else None
            #shutil.copy('../../data/timeloop/hw_.yaml', dir_name+'/hw_.yaml')
            #shutil.copy('../../data/timeloop/mapping_.yaml', dir_name+'/mapping_.yaml')
            #shutil.copy('../../data/timeloop/problem_.yaml', dir_name+'/problem_.yaml')
            #shutil.copy('../../data/timeloop/sparse_.yaml', dir_name+'/sparse_.yaml')
            self.timeloop_estimator.save_timeloop_def(self.model_defs[lr_i], self.best_sol[0], self.best_sol[1][lr_i], 
                                                          dir_name+'/hw_.yaml', dir_name+'/mapping_.yaml', dir_name+'/problem_.yaml', dir_name+'/sparse_.yaml')
        print('best reward: ', total_reward)


    def load_chkpt(self, chkpt):
        raise "Depleted"
        self.reward_rec = chkpt["reward_rec"]
        self.best_reward = chkpt["best_reward"]
        self.best_rewards= chkpt["best_rewards"]
        self.best_sol= chkpt["best_sol"]
        self.worst_reward = chkpt["worst_reward"]
    def get_chkpt(self):
        return  {
            "reward_rec":self.reward_rec,
            "best_rewards": self.best_rewards,
            "best_rewards_iteration": self.best_rewards_iteration,
            "best_sol": self.best_sol,
            "update_best_sol": self.update_best_sol,
            "best_reward": self.best_reward,
            "worst_reward": self.worst_reward,
            "count_invalid": self.count_invalid,
            "start_range": self.start_range,
            "end_range": self.end_range,
        }
    def save_chkpt(self, chkpt_file=None):
        if chkpt_file is None:
            chkpt_file = self.chkpt_file
        chkpt = self.get_chkpt()
        with open(chkpt_file, "wb") as fd:
            pickle.dump(chkpt, fd)
        # print(self.sol)

    def exterior_search(self, gen, layer_info, hw_gene, map_gene, thread_id=None, queue=None):
        if self.fitness == "thrpt_ave" or self.fitness=="thrpt_naive":
            raise "Depleted"
        if self.fitness == "thrpt_btnk":
            raise "Depleted"
        total_reward = 0
        total_constraint = 0
        maestro_state = np.concatenate((hw_gene, map_gene))
        table_entry = tuple(maestro_state)
        if table_entry in self.exp_table:
            reward, constraint = self.exp_table[table_entry]
        else:
            if use_maestro is True:
                reward, constraint = self.oberserve_maestro(layer_info, hw_gene, map_gene)
            else:
                self.observation, (reward, constraint), estimated = self.timeloop_estimator.observe_timeloop(gen, layer_info, hw_gene, map_gene, self.judge, self.fpga_constraint, thread_id=thread_id)
            self.exp_table[table_entry] = (reward, constraint)
        if reward == None:
            queue.put((thread_id, None, None)) if queue is not None else None
            return None, None
        total_reward = reward
        total_constraint = constraint
        #print("estiated: ", total_reward,total_constraint)
        queue.put((thread_id, total_reward, total_constraint)) if queue is not None else None
        return total_reward, total_constraint

