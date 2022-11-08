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
from multiprocessing import Process, Queue


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

    def init_population(self, num_pop, num_layers):
        # Allocate&Initialize VALID population
        hw_new_population = np.empty((num_pop,len(HW_GENE)),dtype=float) # allocation
        map_new_population = np.empty((num_layers, num_pop,len(MAPPING_GENE)),dtype=float) # allocation
        sample_hw_gene = np.array(HWGene.get_sample_gene(), dtype=float).copy()
        sample_map_gene = np.array(MapGene.get_sample_gene(), dtype=float).copy()

        thread_number = 256
        count = 0
        pbar = tqdm(total=num_pop, desc="Initalize New HW Populations")
        while count<num_pop:
            # PE Pop
            reward = None
            constraint = None
            que = Queue()
            threads = []
            rand_gene = []
            for n_ in range(thread_number):
                rand_gene.append(HWGene.generate_random_gene())
                t = Process(target=self.exterior_search, args=(0, self.model_defs[0], rand_gene[n_], sample_map_gene, n_, que))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            for n_ in range(thread_number):
                i_, reward, constraint = que.get()
                if not(reward==None and (constraint==None or constraint > self.constraint_value)) and count<num_pop: #select valid gene. Incorrect, but more score
                #if not(reward==None or (constraint==None or constraint > self.constraint_value)) and count<num_pop: #select valid gene. Correct, but less score
                    hw_new_population[count] = np.array(rand_gene[i_], dtype=float).copy()
                    count += 1
                    pbar.update(1)
        pbar.close()

        thread_number = 256#16*8
        count = 0
        pbar = tqdm(total=num_layers*num_pop, desc="Initalize New Map Populations")
        '''
        while count<num_layers*num_pop:
            # Map Pop for each layers
            reward = None
            que = Queue()
            threads = []
            rand_gene = []
            for n_ in range(thread_number):
                rand_gene.append(MapGene.generate_random_gene())
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
        while count<num_layers*num_pop:
            # Map Pop for each layers
            reward = None
            que = Queue()
            threads = []
            rand_gene = []
            for n_ in range(thread_number):
                rand_gene.append(MapGene.generate_random_gene())
                n_cnt = count+n_//32 if (count+i_//32)<num_layers*num_pop else count
                ly_n = (n_cnt)//num_pop
                h_g_n = (n_cnt)%num_pop
                t = Process(target=self.exterior_search, args=(0, self.model_defs[ly_n], hw_new_population[h_g_n], rand_gene[n_], n_, que))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            for n_ in range(thread_number):
                i_, reward, constraint = que.get()
                if not(reward==None and (constraint==None or constraint > self.constraint_value)) and (count+i_//32)<num_layers*num_pop: #select valid gene
                    map_new_population[(count+i_//32)//num_pop][(count+i_//32)%num_pop] = np.array(rand_gene[i_], dtype=float).copy()
                    #count += 1
                    #pbar.update(1)
            count += 256//32
            pbar.update(256//32)
        pbar.close()
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

    def get_fitness(self, gen, num_pop, num_layers, hw_new_population, map_new_population):
        # get HW/Mapping fitness
        hw_fitness = np.empty(num_pop, float)
        map_fitness = np.empty((num_layers, num_pop), float)
        invalid_count = 0
        pbar = tqdm(total=num_layers*num_pop*num_pop, desc="GA: Get fitness")
        max_num_thread = 48
        n_ = 0
        reward = np.empty((num_pop, num_pop, num_layers), float)
        constraint = np.empty((num_pop, num_pop, num_layers), float)
        '''
        while True:
            que = Queue()
            threads = []
            for th_ in range(max_num_thread):
                i = (n_//num_layers)//num_pop #pop
                j = (n_//num_layers)%num_pop #pop
                l = n_%num_layers #layer
                hw_gene = hw_new_population[i]
                map_gene = map_new_population[l][j]
                t = Process(target=self.exterior_search, args=(gen, self.model_defs[l], hw_gene, map_gene, (i*(num_pop)+j)*num_layers+l, que))
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
                i = (result_n_//num_layers)//num_pop #pop
                j = (result_n_//num_layers)%num_pop #pop
                l = result_n_%num_layers #layer
                reward[i][j][l] = reward_
                constraint[i][j][l] = constraint_
            if n_ >= num_layers*num_pop*num_pop: break
        '''
        for i in range(num_pop):
            for j in range(num_pop):
                for l in range(num_layers):
                    hw_gene = hw_new_population[i]
                    map_gene = map_new_population[l][j]
                    reward_, constraint_ = self.exterior_search(gen, self.model_defs[l], hw_gene, map_gene)
                    if reward_ is None: reward_ = float("-inf")
                    reward[i][j][l] = reward_
                    constraint[i][j][l] = constraint_
                    pbar.update(1)
        pbar.close()
        #get fitness
        '''
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
        '''
        for i in range(num_pop):
            hw_fitness[i] = 0
            for l in range(num_layers):
                hw_fitness[i] += max([reward[i][j][l] for j in range(num_pop)])
        for l in range(num_layers):
            for j in range(num_pop):
                map_fitness[l][j] = max([reward[i][j][l] for i in range(num_pop)])
        print(hw_fitness)
        print("Invalid rate: {:.2f}%({}/{})".format(invalid_count/num_pop*100, invalid_count, num_pop))
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

        # Allocate population & init
        from CMAES_mapper import HWGene_mapper, MapGene_mapper, HWGene_inverse_mapper, MapGene_inverse_mapper

        import pickle
        initial_pop_filename = "initalpop_ours.bin"
        if os.path.exists(initial_pop_filename):
            with open(initial_pop_filename, "rb") as fp:
                source_solutions = pickle.load(fp)
            hw_new_population = np.empty((num_pop,len(HW_GENE)),dtype=float) # allocation
            map_new_population = np.empty((num_layers, num_pop,len(MAPPING_GENE)),dtype=float) # allocation
            for p in range(num_pop):
                hw_new_population[p] = source_solutions[p][0:len(HW_GENE)]
                for l in range(num_layers):
                    map_new_population[l][p] = source_solutions[p][len(HW_GENE)+l*len(MAPPING_GENE):len(HW_GENE)+(l+1)*len(MAPPING_GENE)]
        else:
            # Allocate population & init
            hw_new_population, map_new_population = self.init_population(num_pop, num_layers)
            source_solutions = []
            for p in range(num_pop):
                gad_population = []
                gad_population += hw_new_population[p]
                for l in range(num_layers):
                    gad_population += map_new_population[l][p]
                source_solutions.append(gad_population)
            with open(initial_pop_filename, "wb") as fp:
                pickle.dump(source_solutions, fp)
            logger.debug("[SYSTEM] Generated intial {} population".format(num_pop))

        #hw_new_population, map_new_population = self.init_population(num_pop, num_layers)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hw_fitness, map_fitness = self.get_fitness(0, num_pop, num_layers, hw_new_population, map_new_population)
        source_solutions_hw = []
        source_solutions_map = [[] for lr_i in range(num_layers)]
        for p in range(num_pop):
            source_solutions_hw.append((HWGene_inverse_mapper(hw_new_population[p]), hw_fitness[p]))
            for l in range(num_layers):
                source_solutions_map[l].append((MapGene_inverse_mapper(map_new_population[l][p]), map_fitness[l][p]))

        from cmaes import CMA, SepCMA, get_warm_start_mgd
        ws_mean_hw, ws_sigma_hw, ws_cov_hw = get_warm_start_mgd(source_solutions_hw, gamma=0.1, alpha=0.1)
        ws_mean_map, ws_sigma_map, ws_cov_map = [], [], []
        for l in range(num_layers):
            ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(source_solutions_map[l], gamma=0.1, alpha=0.1)
            ws_mean_map.append(ws_mean)
            ws_sigma_map.append(ws_sigma)
            ws_cov_map.append(ws_cov)

        print(ws_mean, ws_sigma, ws_cov)
        #bounds = np.array([[0, 1] for p in range(len(HW_GENE) + len(MAPPING_GENE)*num_layers)])
        #optimizer = CMA(mean=(np.ones(len(HW_GENE) + len(MAPPING_GENE)*num_layers) / 2), sigma=0.1, bounds=bounds, population_size=num_pop)
        #optimizer = SepCMA(mean=ws_mean, sigma=ws_sigma, cov=ws_cov, bounds=bounds, population_size=num_pop)
        optimizer_hw = CMA(mean=ws_mean_hw, sigma=ws_sigma_hw, cov=ws_cov_hw, bounds=np.array([[0,1] for p in range(len(HW_GENE))]), population_size=num_pop)
        optimizer_map = []
        for l in range(num_layers):
            optimizer_map.append(CMA(mean=ws_mean_map[l], sigma=ws_sigma_map[l], cov=ws_cov_map[l], bounds=np.array([[0,1] for p in range(len(MAPPING_GENE))]), population_size=num_pop))
        logger.debug("[SYSTEM] Generated intial {} population".format(num_pop))
        

        # Get HW&mapping fitness for new pop
        #hw_fitness, map_fitness = self.get_fitness(0, num_pop, num_layers, hw_new_population, map_new_population)
        #print("First fitness optimal : {:9e}".format(min(hw_fitness)))
        #logger.debug("\n\n\n\n")

        # HW -> Mapping Opt.
        iteration = 0

        for generation in range(num_gen):
            print("Gen ", generation)
            #if generation == 10: #?
            #        self.best_reward = float("-inf")
            best_gen_reward =None

            #solutions = []
            num_pop = num_pop
            cma_population = []
            hw_new_population = []
            map_new_population = [[] for i in range(num_layers)]

            for _ in range(num_pop):
                x = optimizer_hw.ask()
                for l in range(num_layers):
                    x = np.concatenate((x, optimizer_map[l].ask()))
                cma_population.append(x)
                hw_new_population.append(HWGene_mapper(x[0:len(HW_GENE)]))
                for l in range(num_layers):
                    map_new_population[l].append(MapGene_mapper(x[len(HW_GENE)+l*len(MAPPING_GENE):len(HW_GENE)+(l+1)*len(MAPPING_GENE)]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hw_fitness, map_fitness = self.get_fitness(generation, num_pop, num_layers, hw_new_population, map_new_population)
            '''
            hw_fitness = []
            while len(cma_population) < num_pop:
                cma_population_sample = []
                hw_new_population = []
                map_new_population = [[] for i in range(num_layers)]
                for _ in range(num_pop):
                    x = optimizer_hw.ask()
                    for l in range(num_layers):
                        x = np.concatenate((x, optimizer_map[l].ask()))
                    cma_population_sample.append(x)
                    hw_new_population.append(HWGene_mapper(x[0:len(HW_GENE)]))
                    for l in range(num_layers):
                        map_new_population[l].append(MapGene_mapper(x[len(HW_GENE)+l*len(MAPPING_GENE):len(HW_GENE)+(l+1)*len(MAPPING_GENE)]))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hw_fitness_sample, map_fitness_sample = self.get_fitness(generation, num_pop, num_layers, hw_new_population, map_new_population)
                for p in range(num_pop):
                    if hw_fitness_sample[p] > float("-1e14"):
                        print(hw_fitness_sample[p])
                        cma_population.append(cma_population_sample[p])
                        hw_fitness.append(hw_fitness_sample[p])
                        map_fitness[l].append(map_fitness_sample[l][p])
                    if len(cma_population) >= num_pop:
                        break
                print("pop inserted {}/{}".format(len(cma_population), num_pop))
            '''
            solutions_hw = []
            solutions_map = [[] for i in range(num_layers)]
            for p in range(num_pop):
                solutions_hw.append((cma_population[p][0:len(HW_GENE)], hw_fitness[p]))
                for l in range(num_layers):
                    solutions_map[l].append((cma_population[p][len(HW_GENE)+l*len(MAPPING_GENE):len(HW_GENE)+(l+1)*len(MAPPING_GENE)], map_fitness[l][p]))
            optimizer_hw.tell(solutions_hw)
            for l in range(num_layers):
                optimizer_map[l].tell(solutions_map[l])
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
            best_tmp_results = self.timeloop_estimator.get_gene_HW_info(self.model_defs[0], self.best_sol[0], self.best_sol[1][0])
            print("Best sol(Xdim, X, Ydim, Y, group_density, bank): ", best_tmp_results)
            if best_gen_reward  is not None:
                logger.debug("\nHW Generation {}: new best award reward: {:9e}".format(generation+1, self.best_reward))
            self.save_chkpt()
        # == end of HW GA

        self.save_chkpt()
        logger.debug("Best result total:")
        logger.debug(self.best_rewards)
        logger.debug("Constraints of best")
        logger.debug(self.best_rewards_constraint)

        import shutil
        outdir = self.outdir #'../../data/best'
        os.mkdir(outdir) if os.path.exists(outdir) is False else None
        total_reward = 0
        print("best reward each layer")
        for lr_i in range(num_layers):
            reward, constraint = self.exterior_search(num_gen-1, self.model_defs[lr_i], self.best_sol[0], self.best_sol[1][lr_i])
            print(reward)
            total_reward += reward
            dir_name = outdir+'/{:02d}'.format(lr_i)
            os.mkdir(dir_name) if os.path.exists(dir_name) is False else None
            self.timeloop_estimator.save_timeloop_def(self.model_defs[lr_i], self.best_sol[0], self.best_sol[1][lr_i], 
                                                          dir_name+'/hw_.yaml', dir_name+'/mapping_.yaml', dir_name+'/problem_.yaml', dir_name+'/sparse_.yaml')
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

