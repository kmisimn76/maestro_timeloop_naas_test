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
#from bayes_opt import BayesianOptimization
#from scipy import optimize
from multiprocessing import Process, Queue, Manager, Lock

use_maestro = False
use_sparseloop = False #True


action_space, action_bound, action_bottom = get_action_space()
action_space = [act * bound for act, bound in zip(action_space, action_bound)]
start_range, end_range = 0, len(action_space[0])

from HWGene import _HWGene, HW_GENE
HWGene = _HWGene()
from MapGene import _MapGene, MAPPING_GENE
MapGene = _MapGene()

from TimeloopEstimation import TimeloopEstimator
from SparseloopEstimation import SparseloopEstimator
from SparseAccelEstimation import SparseAccelEstimator
from FPGAConstraint.AlveoU200 import Constraint_AlveoU200_Sparse

class NAAS(object):


    def __init__(self, model_defs, outdir, epoch_info, finish_reward=100, dim_size=6,n_action_steps=2, resource_size=2, dataflow="dla", is_discrete=True):
        super(NAAS,self).__init__()
        dst_path = "../../cost_model/maestro"

        self.outdir = outdir

        self.epochs = epoch_info['epochs'] #7000#3600 #hardcoded
        self.num_pop = epoch_info['num_pop'] #100#60 #hardcoded
        self.num_gen = epoch_info['num_gen'] #epochs // num_pop
        self.num_parents = epoch_info['num_parents'] #40#30 #hardcoded

        maestro = dst_path
        self._executable = "{}".format(maestro)
        random.seed()
        random_file_name = random.randint(0, 2 ** 31)
        self.random_file_name = "{}".format(random_file_name)

        self.fpga_constraint = Constraint_AlveoU200_Sparse()
        #self.timeloop_estimator = SparseloopEstimator(self.random_file_name) if use_sparseloop else TimeloopEstimator(self.random_file_name)
        self.timeloop_estimator = SparseAccelEstimator(self.random_file_name) #FIXME:
        print("Use SparseAccelEstimator, under test")

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

    def init_population_worker(self, lock, hw_pop_pool, num_pop, num_layers):
        while len(hw_pop_pool) < num_pop:
            fit = float("-inf")
            last_pop = HWGene.generate_random_gene()
            last_cnt = 0
            while fit < float("-1e14"):
                hw_new_pop = HWGene.generate_random_gene()
                hw_fitness, map_fitness, _ = self.get_fitness(0, 1, num_layers, [hw_new_pop], [[MapGene.generate_random_gene()] for _ in range(num_layers)], verbose=False)
                last_cnt = last_cnt+1 if (sum([0 if last_pop[i]==hw_new_pop[i] else 1 for i in range(len(hw_new_pop))]) == 0) else 0
                last_pop = hw_new_pop
                if max(map_fitness) > float("-1e14") or (last_cnt > 5):
                    break
            print("!")
            lock.acquire()
            hw_pop_pool.append(hw_new_pop)
            lock.release()


    def init_population(self, num_pop, num_layers):
        # Allocate&Initialize VALID population
        hw_new_population = np.empty((num_pop,len(HW_GENE)),dtype=float) # allocation
        map_new_population = np.empty((num_layers, num_pop,len(MAPPING_GENE)),dtype=float) # allocation
        sample_hw_gene = np.array(HWGene.get_sample_gene(), dtype=float).copy()
        sample_map_gene = np.array(MapGene.get_sample_gene(), dtype=float).copy()

        from time import time
        '''
        for i in range(num_pop):
            fit = float("-inf")
            while fit < float("-1e14"):
                hw_new_population[i] = HWGene.generate_random_gene()
                start = time()
                #for it in range(100):
                #    hw_fitness, _, _ = self.get_fitness(0, 1, 1, [hw_new_population[i]], [[MapGene.get_sample_gene(seed1=(it+1)/100, seed2=0.01)]], specific_layer=random.randint(1,num_layers-1), verbose=False)
                #    #hw_fitness, _, _ = self.get_fitness(0, 1, 1, [hw_new_population[i]], [[MapGene.generate_random_gene()]], verbose=False)
                #    fit = max(fit, hw_fitness[0])
                #    if fit > float("-1e14"): break
                hw_fitness, map_fitness, _ = self.get_fitness(0, 1, num_layers, [hw_new_population[i]], [[MapGene.generate_random_gene()] for _ in range(num_layers)], verbose=False)
                if max(map_fitness) > float("-1e14"):
                    break
                end = time()
                #print((end - start)*1000)
            print("!")
        '''
        threads = []
        num_thread = 48
        manager = Manager()
        hw_pop_pool = manager.list()
        lock = Lock()
        for n in range(num_thread):
            t = Process(target=self.init_population_worker, args=( lock, hw_pop_pool, num_pop, num_layers))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        for i in range(num_pop):
            hw_new_population[i] = hw_pop_pool[i]

        for i in range(num_pop):
            for l in range(num_layers):
                map_new_population[l][i] = MapGene.generate_random_gene()

        print("Setting initial populations")

        pbar = tqdm(total=(1+num_layers)*num_pop, desc="Initalize New HW Populations")

        invalid_count = num_pop*(1+num_layers)
        while invalid_count > (1+num_layers)*num_pop//10: # 90% pop must satify constraint
            hw_fitness, map_fitness, _ = self.get_fitness(0, num_pop, num_layers, hw_new_population, map_new_population, verbose=False)
            invalid_count = 0
            '''
            for i in range(num_pop):
                if hw_fitness[i] is None or hw_fitness[i] <= float("-1e14"):
                    invalid_count += 1
                    hw_new_population[i] = HWGene.generate_random_gene()
            '''
            for j in range(num_pop):
                for l in range(num_layers):
                    if map_fitness[l][j] is None or map_fitness[l][j] <= float("-1e14"):
                        invalid_count += 1
                        map_new_population[l][j] = MapGene.generate_random_gene()
            pbar.n = num_pop*(1+num_layers) - invalid_count
            pbar.refresh()
            if num_pop*(1+num_layers) - invalid_count == 0:
                print("initialization failed. please re run")
                exit()
        pbar.close()

        print("Initial pops: ", hw_new_population)

        return hw_new_population, map_new_population

    def get_fitness_(self, gen, num_pop, num_layers, hw_new_population, map_new_population):
        # get HW/Mapping fitness
        hw_fitness = np.empty(num_pop, float)
        map_fitness = np.empty((num_layers, num_pop), float)
        invalid_count = 0
        pbar = tqdm(total=num_layers*num_pop, desc="GA: Get fitness")
        max_num_thread = 48
        n_ = 0
        reward = np.empty((num_pop, num_layers), float)
        constraint = np.empty((num_pop, num_layers), float)
        while True:
            que = Queue()
            threads = []
            for th_ in range(max_num_thread):
                i = n_//num_layers #pop
                j = n_%num_layers #layer
                hw_gene = hw_new_population[i]
                map_gene = map_new_population[j][i]
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
                i = result_n_//num_layers
                j = result_n_%num_layers
                reward[i][j] = reward_
                constraint[i][j] = constraint_
            if n_ >= num_layers*num_pop: break
        pbar.close()
        #get fitness
        for i in range(num_pop):
            tot_reward = 0
            tot_constraint = 0
            for j in range(num_layers):
                map_fitness[j][i] = reward[i][j] #FIXME
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
            #for j in range(num_layers): #FIXME
                #map_fitness[j][i] = hw_fitness[i] #FIXME
        print("Invalid rate: {:.2f}%({}/{})".format(invalid_count/num_pop*100, invalid_count, num_pop))
        return hw_fitness, map_fitness


    def get_fitness_proc(self, gen, num_pop, num_layers, hw_new_population, map_new_population, n, num_hw_pop, reward, constraint, return_dict, pbar,verbose=True):
        for i in range(n*num_hw_pop, min(num_pop, (n+1)*num_hw_pop)):
            for j in range(num_pop):
                for l in range(num_layers):
                    hw_gene = hw_new_population[i]
                    map_gene = map_new_population[l][j]
                    reward_, constraint_ = self.exterior_search(gen, self.model_defs[l], hw_gene, map_gene)
                    if reward_ is None: reward_ = float("-inf")
                    reward[i][j][l] = reward_
                    constraint[i][j][l] = constraint_
                    if verbose and n==0:
                        pbar.update(math.ceil(num_pop/num_hw_pop))
        return_dict[n] = (reward, constraint)

    def get_fitness(self, gen, num_pop, num_layers, hw_new_population, map_new_population, specific_layer=None, verbose=True):
        # get HW/Mapping fitness
        hw_fitness = np.empty(num_pop, float)
        map_fitness = np.empty((num_layers, num_pop), float)
        invalid_count = 0
        if verbose:
            pbar = tqdm(total=num_layers*num_pop*num_pop, desc="GA: Get fitness")
        else:
            pbar = None
        n_ = 0
        reward = np.empty((num_pop, num_pop, num_layers), float)
        constraint = np.empty((num_pop, num_pop, num_layers), float)

        if num_pop < 10:
            for i in range(num_pop):
                for j in range(num_pop):
                    for l in range(num_layers):
                        hw_gene = hw_new_population[i]
                        map_gene = map_new_population[l][j]
                        reward_, constraint_ = self.exterior_search(gen, self.model_defs[l if specific_layer is None else specific_layer], hw_gene, map_gene)
                        if reward_ is None: reward_ = float("-inf")
                        reward[i][j][l] = reward_
                        constraint[i][j][l] = constraint_
                        if verbose:
                            pbar.update(1)
        else:
            threads = []
            num_thread = 25
            manager = Manager()
            return_dict = manager.dict()
            #for n in range(math.ceil(num_pop/num_thread)):
            for n in range(num_thread):
                t = Process(target=self.get_fitness_proc, args=(gen, num_pop, num_layers, hw_new_population, map_new_population, n, math.ceil(num_pop/num_thread), reward, constraint, return_dict, pbar, verbose))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

            for n in range(num_thread):
                reward_dict = return_dict[n][0]
                reward_const = return_dict[n][1]
                for i in range(n*(math.ceil(num_pop/num_thread)), min(num_pop, (n+1)*(math.ceil(num_pop/num_thread)))):
                    for j in range(num_pop):
                        for l in range(num_layers):
                            reward[i][j][l] = reward_dict[i][j][l]
                            constraint[i][j][l] = reward_const[i][j][l]
        if verbose:
            pbar.close()

        #get fitness
        invalid_count = 0
        for i in range(num_pop):
            hw_fitness[i] = 0
            for l in range(num_layers):
                hw_fitness[i] += max([reward[i][j][l] for j in range(num_pop)])
            if hw_fitness[i] is None or hw_fitness[i] <= float("-1e14"):
                invalid_count += 1
        for l in range(num_layers):
            for j in range(num_pop):
                map_fitness[l][j] = max([reward[i][j][l] for i in range(num_pop)])
        if verbose:
            hw_valid_fitness = [hw_fitness[i] for i in range(num_pop)]
            hw_valid_fitness = np.array(list(filter(lambda x: x > float("-1e14"), hw_valid_fitness)))
            print("fitness result")
            print("- HW fitness info(min, max, mean, var, # valid): ", "{:2e} {:2e} {:2e} {:2e} {}".format(min(hw_valid_fitness), max(hw_valid_fitness), hw_valid_fitness.mean(), hw_valid_fitness.var()**0.5, np.sum(hw_fitness != float("-inf"))))
            print("- Invalid rate: {:.2f}%({}/{})".format(invalid_count/num_pop*100, invalid_count, num_pop))

        with open("design_points.txt", "a") as f:
            for i in range(num_pop):
                optmappings = []
                opthw = []
                optimal = 0
                hw_gene = hw_new_population[i]
                opthw = [g for g in hw_gene]
                for l in range(num_layers):
                    optmapping = None
                    optfit = float("-inf")
                    for j in range(num_pop):
                        map_gene = map_new_population[l][j]
                        if optfit <= reward[i][j][l]:
                            optmapping = [g for g in map_gene]
                            optfit = reward[i][j][l]
                    optmappings.append(optmapping)
                    optimal += optfit
                if optimal != None and optimal >= float("-1e14"):
                    outstr = [str(optimal)] + [(",".join([str(d) for d in opthw]))] + [",".join([str(d) for d in optm]) for optm in optmappings]
                    f.write(",".join(outstr)+"\n")

        return hw_fitness, map_fitness, reward



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


        epochs = self.epochs #7000#3600 #hardcoded
        num_pop = self.num_pop #100#60 #hardcoded
        num_gen = self.num_gen #epochs // num_pop
        num_parents = self.num_parents #40#30 #hardcoded
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

        from PyGAD_mapper import HWGene_mapper, MapGene_mapper, HWGene_inverse_mapper, MapGene_inverse_mapper

        import pickle
        #initial_pop_filename = "initalpop_ours.bin"
        initial_pop_filename = "initalpop_ours_fp.bin"
        if os.path.exists(initial_pop_filename):
            with open(initial_pop_filename, "rb") as fp:
                source_solutions = pickle.load(fp)
            hw_new_population = np.empty((num_pop,len(HW_GENE)),dtype=float) # allocation
            map_new_population = np.empty((num_layers, num_pop,len(MAPPING_GENE)),dtype=float) # allocation
            for p in range(num_pop):
                hw_new_population[p] = HWGene_mapper(source_solutions[p][0:len(HW_GENE)])
                for l in range(num_layers):
                    map_new_population[l][p] = MapGene_mapper(source_solutions[p][len(HW_GENE)+l*len(MAPPING_GENE):len(HW_GENE)+(l+1)*len(MAPPING_GENE)])
        else:
            # Allocate population & init
            hw_new_population, map_new_population = self.init_population(num_pop, num_layers)
            source_solutions = []
            for p in range(num_pop):
                gad_population = []
                gad_population += HWGene_inverse_mapper(hw_new_population[p])
                for l in range(num_layers):
                    gad_population += MapGene_inverse_mapper(map_new_population[l][p])
                source_solutions.append(gad_population)
            with open(initial_pop_filename, "wb") as fp:
                pickle.dump(source_solutions, fp)
            logger.debug("[SYSTEM] Generated intial {} population".format(num_pop))


        

        # Get HW&mapping fitness for new pop
        hw_fitness, map_fitness, _ = self.get_fitness(0, num_pop, num_layers, hw_new_population, map_new_population)
        print("First fitness optimal : {:9e}".format(max(hw_fitness)))
        logger.debug("\n\n\n\n")

        # HW -> Mapping Opt.
        iteration = 0

        for generation in range(num_gen):
            print("Gen ", generation)
            #if generation == 10: #?
            #        self.best_reward = float("-inf")
            best_gen_reward =None
            hw_parents = HWGene.select_parents(hw_new_population, hw_fitness, num_parents, gen=generation)
            map_parents = [[] for lr_i in range(num_layers)]
            for lr_i in range(num_layers):
                map_parents[lr_i] = MapGene.select_parents(map_new_population[lr_i], map_fitness[lr_i], num_parents)

            hw_offspring_mutation = []
            map_offspring_mutation = [[] for lr_i in range(num_layers)]
            
            while len(hw_offspring_mutation) < num_pop-num_parents: # gather valid child
                hw_offspring_crossover_sample = HWGene.crossover(hw_parents,
                                                num_pop-num_parents)
                hw_offspring_mutation_sample = HWGene.mutation(hw_offspring_crossover_sample, rate=0.08)
    
                map_offspring_mutation_sample = [[] for lr_i in range(num_layers)]
                for lr_i in range(num_layers):
                    map_offspring_crossover_sample = MapGene.crossover(map_parents[lr_i],
                                                            num_pop-num_parents)
                    map_offspring_mutation_sample[lr_i] = MapGene.mutation(map_offspring_crossover_sample, rate=0.08)
                #hw_fitness_sample, map_fitness_sample = self.get_fitness(generation, num_pop-num_parents, num_layers, hw_offspring_mutation_sample, map_offspring_mutation_sample)
                hw_fitness_sample, map_fitness_sample = [0 for p in range(num_pop)], []

                for p in range(num_pop-num_parents):
                    if hw_fitness_sample[p] > float("-1e14") or True:
                        #print(hw_fitness_sample[p])
                        hw_offspring_mutation.append(hw_offspring_mutation_sample[p])
                        for lr_i in range(num_layers):
                            map_offspring_mutation[lr_i].append(map_offspring_mutation_sample[lr_i][p])
                    if len(hw_offspring_mutation) >= num_pop-num_parents:
                        break

            hw_new_population[0:hw_parents.shape[0], :] = hw_parents
            hw_new_population[hw_parents.shape[0]:, :] = hw_offspring_mutation
            for lr_i in range(num_layers):
                map_new_population[lr_i][0:map_parents[0].shape[0], :] = map_parents[lr_i]
                map_new_population[lr_i][map_parents[0].shape[0]:, :] = map_offspring_mutation[lr_i]

            hw_fitness, map_fitness, reward_fit = self.get_fitness(generation, num_pop, num_layers, hw_new_population, map_new_population)

            for pop in range(num_pop):
                reward = hw_fitness[pop]
                if reward > self.best_reward:
                    best_gen_reward = reward
                    self.best_reward = reward
                    self.best_reward_constraint = float("inf") #total_used_constraint/num_layers
                    self.best_sol = (hw_new_population[pop].copy(), [map_new_population[lr_i][np.argmax([reward_fit[pop][map_pop][lr_i] for map_pop in range(num_pop)])].copy() for lr_i in range(num_layers)])
                iteration += 1
                self.best_rewards_iteration.append(iteration)
                self.best_rewards_constraint.append(self.best_reward_constraint)
                self.best_rewards.append(self.best_reward)
                self.save_chkpt()
            best_tmp_results = self.timeloop_estimator.get_gene_HW_info(self.model_defs[0], self.best_sol[0], self.best_sol[1][0])
            print("Best sol(Xdim, X, Ydim, Y, Zdim, Z, group_density, bank, buffers, use_sparsity): ", best_tmp_results)
            if best_gen_reward  is not None:
                logger.debug("\nHW Generation {}: new best award reward: {:9e}".format(generation+1, self.best_reward))
        # == end of HW GA

        total_reward = self.save_chkpt(verbose=True)
        #logger.debug("Best result total:")
        #logger.debug(self.best_rewards)
        #logger.debug("Constraints of best")
        #logger.debug(self.best_rewards_constraint)


        print('best reward: ', total_reward)
        print('raw mapping data stored @ {}'.format(dir_name+'/hw_.yaml'))


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
    def save_chkpt(self, chkpt_file=None, verbose=False):
        if chkpt_file is None:
            chkpt_file = self.chkpt_file
        chkpt = self.get_chkpt()
        with open(chkpt_file, "wb") as fd:
            pickle.dump(chkpt, fd)
        # print(self.sol)
        import shutil
        outdir = self.outdir #'../../data/best'
        os.mkdir(outdir) if os.path.exists(outdir) is False else None
        total_reward = 0
        num_layers = len(self.model_defs)
        for lr_i in range(num_layers):
            reward, constraint = self.exterior_search(0, self.model_defs[lr_i], self.best_sol[0], self.best_sol[1][lr_i])
            if verbose:
                print(reward)
                total_reward += reward
            else:
                total_reward += reward
                dir_name = outdir+'/{:02d}'.format(lr_i)
                os.mkdir(dir_name) if os.path.exists(dir_name) is False else None
                self.timeloop_estimator.save_timeloop_def(self.model_defs[lr_i], self.best_sol[0], self.best_sol[1][lr_i], 
                                                              dir_name+'/hw_.yaml', dir_name+'/mapping_.yaml', dir_name+'/problem_.yaml', dir_name+'/sparse_.yaml')
        return total_reward

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

