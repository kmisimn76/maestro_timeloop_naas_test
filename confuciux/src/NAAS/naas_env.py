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


action_space, action_bound, action_bottom = get_action_space()
action_space = [act * bound for act, bound in zip(action_space, action_bound)]
start_range, end_range = 0, len(action_space[0])
df_dict = {1:"dla", 2:"shi", 3:"eye"}
m_type_dicts = {1:"CONV", 2:"DSCONV"}


#from enum import Enum
from enum import IntEnum
class HW_GENE (IntEnum):
    L2_SIZE     = 0
    L1_SIZE     = 1
    NUM_PE      = 2
    BW          = 3
    NUM_DIM     = 4
    DIM_SIZE_0  = 5
    DIM_SIZE_1  = 6
    DIM_SIZE_2  = 7
    PAR_DIM_K   = 8
    PAR_DIM_C   = 9
    PAR_DIM_Y   = 10
    PAR_DIM_X   = 11
    PAR_DIM_R   = 12
    PAR_DIM_S   = 13
class MAPPING_GENE (IntEnum):
    ARR_TILE_SIZE_K  = 6
    ARR_TILE_SIZE_C  = 7
    ARR_TILE_SIZE_Y  = 8
    ARR_TILE_SIZE_X  = 9
    ARR_TILE_SIZE_R  = 10
    ARR_TILE_SIZE_S  = 11
    ARR_LOOP_ORDER_K = 0
    ARR_LOOP_ORDER_C = 1
    ARR_LOOP_ORDER_Y = 2
    ARR_LOOP_ORDER_X = 3
    ARR_LOOP_ORDER_R = 4
    ARR_LOOP_ORDER_S = 5
    PE_LOOP_ORDER_K  = 12
    PE_LOOP_ORDER_C  = 13
    PE_LOOP_ORDER_Y  = 14
    PE_LOOP_ORDER_X  = 15
    PE_LOOP_ORDER_R  = 16
    PE_LOOP_ORDER_S  = 17
class DIM (IntEnum):
    K = 0
    C = 1
    Y = 2
    X = 3
    R = 4
    S = 5
    P = 6
    def to_str(data):
        if data==DIM.K: return "K"
        if data==DIM.C: return "C"
        if data==DIM.Y: return "Y"
        if data==DIM.X: return "X"
        if data==DIM.R: return "R"
        if data==DIM.S: return "S"
        if data==DIM.P: return "P"
        print(data)
        raise "Exception"
class MAESTRO_DATAFLOW:
    def __init__(self):
        self.mapping = []
    def length(self):
        return len(self.mapping)
    def read_line(self, i):
        return self.mapping[i][0], DIM.to_str(self.mapping[i][1]), self.mapping[i][2], self.mapping[i][3]
    def push_temporal_outer(self, var, size):
        self.mapping.append(["Temporal", var, size, size])
    def push_spatial_outer(self, var, size):
        self.mapping.append(["Spatial", var, size, size])
    def push_temporal_inner(self, var):
        self.mapping.append(["Temporal", var, 1, 1])
    def push_spatial_inner(self, var):
        self.mapping.append(["Spatial", var, 1, 1])
    def push_cluster(self, size):
        self.mapping.append(["Cluster", DIM.P, size, size])

class MyBounds(object, ):
    def __init__(self,length):
        xmax = [11 for _ in range(length)]
        xmin = [0 for _ in range(length)]
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

def select_parents(pop, fitness, num_parents, num_layers):
    raise "Deplete"
    parents = np.empty((num_parents, num_layers, 2))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        print(max_fitness_idx)
        print(pop.shape)
        parents[parent_num] = pop[max_fitness_idx]
        fitness[max_fitness_idx] = float("-Inf")
    return parents

def crossover(parents, offspring_size, num_layers):
    raise "Deplete"
    offspring = np.empty((offspring_size, num_layers, 2))
    crossover_point = np.uint8(num_layers/2)

    for k in range(offspring_size):
        parent1_idx = k%parents.shape[0]
        parent2_idx = np.random.randint(0, parents.shape[0]) #(k+1)%parents.shape[0]
        offspring[k][0:crossover_point] = parents[parent1_idx][0:crossover_point]
        offspring[k][crossover_point:] = parents[parent2_idx][crossover_point:]
    return offspring

def mutation(offsprings, num_layers,rate=0.05):
    raise "Deplete"
    for idx in range(offsprings.shape[0]):
        for lay in range(offsprings.shape[1]):
            for p in range(offsprings.shape[2]):
                if random.random() < rate:
                    offsprings[idx][lay][p] = random.randint(0, 11)
    return offsprings

class _HWGene(object):
    def get_sample_gene(self):
        return [3000, 100, 256, 50, 2, 0.5, 0.5, 0.5, 6,5,4,3,2,1]
    def generate_random_gene(self):
        L2Buf = random.randint(0, 1000)
        L1Buf = random.randint(0, 1000)
        PEs = random.randint(0, 16*16)
        BW = random.randint(0,1000)
        NumDim = random.randint(2,2) #FIXME : Hardcoding, Fix NumDim to 2
        #DimSize0 = random.randint(1,16*16)
        #DimSize1 = random.randint(1,16*16)
        #DimSize2 = random.randint(1,16*16)
        DimSize0 = random.random() #0~1 -> to softmax
        DimSize1 = random.random() #0~1
        DimSize2 = random.random() #0~1
        ParDimK = random.randint(1,6)
        ParDimC = random.randint(1,6)
        ParDimY = random.randint(1,6)
        ParDimX = random.randint(1,6)
        ParDimR = random.randint(1,6)
        ParDimS = random.randint(1,6)
        return [L2Buf, L1Buf, PEs, BW, NumDim,
                    DimSize0, DimSize1, DimSize2,
                    ParDimK, ParDimC, ParDimY, ParDimX, ParDimR, ParDimS]

    #HWGene
    def select_parents(self, pop, fitness, num_parents):
        parents = np.empty((num_parents, 14))
        for parent_num in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num] = pop[max_fitness_idx]
            fitness[max_fitness_idx] = float("-Inf")
        return parents

    #HWGene
    def crossover(self, parents, offspring_size):
        offspring = np.empty((offspring_size, 14))
        #crossover_point = np.uint8(14/2)
        for k in range(offspring_size):
            parent1_idx = k%parents.shape[0]
            parent2_idx = np.random.randint(0, parents.shape[0]) #(k+1)%parents.shape[0]
            #offspring[k][0:crossover_point] = parents[parent1_idx][0:crossover_point]
            #offspring[k][crossover_point:] = parents[parent2_idx][crossover_point:]
            for i in range(14):
                offspring[k][i] = parents[parent1_idx][i] if random.randint(0,1)==0 else parents[parent2_idx][i] #crossover 1:1
        return offspring

    #HWGene
    def mutation(self, offsprings,rate=0.05):
        rand_list = self.generate_random_gene()
        for idx in range(offsprings.shape[0]):
            for p in range(offsprings.shape[1]):
                if random.random() < rate:
                    offsprings[idx][p] = rand_list[p]
        return offsprings
HWGene = _HWGene()

class _MapGene(object):
    def get_sample_gene(self):
        return [0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                6,5,4,3,2,1, 6,5,4,3,2,1]
    def generate_random_gene(self):
        #ArrTileSizeK = random.randint(1,min(16, self.model_defs[i][0]))
        #ArrTileSizeC = random.randint(1,min(16, self.model_defs[i][1]))
        #ArrTileSizeY = random.randint(1,min(16, self.model_defs[i][2]))
        #ArrTileSizeX = random.randint(1,min(16, self.model_defs[i][3]))
        #ArrTileSizeR = random.randint(1,min(16, self.model_defs[i][4]))
        #ArrTileSizeS = random.randint(1,min(16, self.model_defs[i][5]))
        ArrTileSizeK = random.random() #0~1
        ArrTileSizeC = random.random()
        ArrTileSizeY = random.random()
        ArrTileSizeX = random.random()
        ArrTileSizeR = random.random()
        ArrTileSizeS = random.random()
        ArrLoopOrderK = random.randint(1, 6)
        ArrLoopOrderC = random.randint(1, 6)
        ArrLoopOrderY = random.randint(1, 6)
        ArrLoopOrderX = random.randint(1, 6)
        ArrLoopOrderR = random.randint(1, 6)
        ArrLoopOrderS = random.randint(1, 6)
        PELoopOrderK = random.randint(1, 6)
        PELoopOrderC = random.randint(1, 6)
        PELoopOrderY = random.randint(1, 6)
        PELoopOrderX = random.randint(1, 6)
        PELoopOrderR = random.randint(1, 6)
        PELoopOrderS = random.randint(1, 6)
        return [
                    ArrTileSizeK,ArrTileSizeC,ArrTileSizeY,ArrTileSizeX,ArrTileSizeR,ArrTileSizeS,
                    ArrLoopOrderK,ArrLoopOrderC,ArrLoopOrderY,ArrLoopOrderX,ArrLoopOrderR,ArrLoopOrderS,
                    PELoopOrderK,PELoopOrderC,PELoopOrderY,PELoopOrderX,PELoopOrderR,PELoopOrderS
                ]

    #MapGene
    def select_parents(self, pop, fitness, num_parents):
        parents = np.empty((num_parents, 18))
        for parent_num in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num] = pop[max_fitness_idx]
            fitness[max_fitness_idx] = float("-Inf")
        return parents

    #MapGene
    def crossover(self, parents, offspring_size):
        offspring = np.empty((offspring_size, 18))
        for k in range(offspring_size):
            parent1_idx = k%parents.shape[0]
            parent2_idx = np.random.randint(0, parents.shape[0]) #(k+1)%parents.shape[0]
            #offspring[k][0:crossover_point] = parents[parent1_idx][0:crossover_point]
            #offspring[k][crossover_point:] = parents[parent2_idx][crossover_point:]
            for i in range(18):
                offspring[k][i] = parents[parent1_idx][i] if random.randint(0,1)==0 else parents[parent2_idx][i] #crossover 1:1
        return offspring

    #MapGene
    def mutation(self, offsprings,rate=0.05):
        rand_list = self.generate_random_gene()
        for idx in range(offsprings.shape[0]):
            for p in range(offsprings.shape[1]):
                if random.random() < rate:
                    offsprings[idx][p] = rand_list[p]
        return offsprings
MapGene = _MapGene()

class MaestroEnvironment(object):


    def __init__(self, model_defs, finish_reward=100, dim_size=6,n_action_steps=2, resource_size=2, dataflow="dla", is_discrete=True):
        super(MaestroEnvironment,self).__init__()
        dst_path = "../../cost_model/maestro"

        maestro = dst_path
        self._executable = "{}".format(maestro)
        random.seed()
        random_file_name = random.randint(0, 2 ** 31)
        self.random_file_name = "{}".format(random_file_name)
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
        sol = [bound[:self.n_action_steps] for i in range(len(self.model_defs))]
        _, total_constraint = self.exterior_search(sol)
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

    def judge(self):
        runtime, throughput, energy, area, l1_size, l2_size, mac, power = self.observation
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


    def genetic_search(self, epochs=100, chkpt_file="genetic_chkpt.plt",fd=None):

        epochs = 30000 #hardcoded
        num_pop = 30 #hardcoded
        num_gen = epochs // num_pop
        num_parents = 15 #hardcoded
        self.fd = fd
        self.chkpt_file = chkpt_file
        self.start_range = start_range
        self.end_range = end_range-1

        if self.best_reward is None:
            self.best_reward = float("-Inf")
        self.best_rewards = []
        for i in range(len(self.model_defs)):
            if self.map_best_reward[i] is None:
                self.map_best_reward[i] = float("-Inf")
        self.epoch = 0
        self.best_sol = None
        num_layers = len(self.model_defs)

        '''
        df_example = MAESTRO_DATAFLOW()
        df_example.push_temporal_outer(DIM.R, 3)
        df_example.push_temporal_outer(DIM.S, 3)
        df_example.push_temporal_outer(DIM.C, 4)
        df_example.push_temporal_outer(DIM.Y, 4)
        df_example.push_temporal_outer(DIM.X, 4)
        df_example.push_spatial_outer(DIM.K, 1)
        df_example.push_cluster(16)
        df_example.push_temporal_inner(DIM.R)
        df_example.push_temporal_inner(DIM.S)
        df_example.push_temporal_inner(DIM.K)
        df_example.push_temporal_inner(DIM.Y)
        df_example.push_temporal_inner(DIM.X)
        df_example.push_spatial_inner(DIM.C)
        
        self.write_maestro(self.model_defs[0], dataflow=df_example, m_file="temporal")
        '''

        self.num_generations = num_gen
        self.num_population = num_pop
        self.num_parents = num_parents

        print("Number of generations: " + str(num_gen))
        print("Number of population: " + str(num_pop))
        print("Number of parents: " + str(num_parents))

        #L2 buf size, L1 buf size, # PEs, L2-L1 BW, #Dim, Dim size (3), ParDim (6)
        hw_new_population = np.empty((num_pop,14),dtype=float)
        map_new_population = np.empty((num_layers, num_pop,18),dtype=float)
        #guess_action = np.empty((num_layers,2 ))
        
        sample_hw_gene = np.array(HWGene.get_sample_gene(), dtype=float).copy()
        sample_map_gene = np.array(MapGene.get_sample_gene(), dtype=float).copy()
        #Initialize Valid Population
        for count in tqdm(range(num_pop), desc="Initalize New Populations"):
            # PE Pop
            reward = None
            constraint = None
            while reward==None and (constraint==None or constraint > self.constraint_value):
                rand_gene = HWGene.generate_random_gene()
                hw_new_population[count] = np.array(rand_gene, dtype=float).copy()
                reward, constraint = self.oberserve_maestro(self.model_defs[0], hw_new_population[count], sample_map_gene)
            # Map Pop
            for i in range(num_layers):
                reward = None
                while reward==None:
                    rand_gene = MapGene.generate_random_gene()
                    map_new_population[i][count] = np.array(rand_gene, dtype=float).copy()
                    reward, constraint = self.oberserve_maestro(self.model_defs[0], sample_hw_gene, map_new_population[i][count])

            count += 1
        print("[SYSTEM] Generated intial {} population".format(num_pop))
        
        # get HW fitness
        hw_fitness = np.empty(num_pop, float)
        invalid_count = 0
        for i in tqdm(range(num_pop), desc="HW initial fitness"):
            hw_gene = hw_new_population[i]
            reward,total_used_constraint  = self.exterior_search_all_layer(self.model_defs, hw_gene, map_new_population, 0)
            if reward is None: # Can't compilation
                reward = float("-Inf")
                #print("Error with reward")
                #print(new_population[i])
                #exit(-1)
                invalid_count += 1
            elif total_used_constraint > self.constraint_value:
                reward = float("-Inf")
                invalid_count += 1
            hw_fitness[i] = reward
        print("Invalid rate: {:.2f}%({}/{})".format(invalid_count/num_pop*100, invalid_count, num_pop))
        # get Mapping fitness
        print("\n\n\n")
        map_fitness = np.empty((num_layers, num_pop), float)
        layer_bar = tqdm(total=num_layers, desc="Layer progress in Mapping initial fitness",position=0)
        bar_log = tqdm(total=0, bar_format='{desc}', position=2)
        pop_bar = tqdm(total=num_pop, position=1)
        for layer in range(num_layers):
            hw_gene = hw_new_population[0]
            invalid_count = 0
            pop_bar.set_description_str(desc="Mapping initial fitness layer {}".format(layer))
            for i in range(num_pop):
                map_gene = map_new_population[layer][i]
                reward,total_used_constraint  = self.exterior_search(self.model_defs[layer], hw_gene, map_gene)
                if reward is None:
                    reward = float("-Inf")
                    #print("Error with reward")
                    #print(new_population[i])
                    #exit(-1)
                    invalid_count += 1
                elif total_used_constraint > self.constraint_value:
                    reward = float("-Inf")
                    invalid_count += 1
                map_fitness[layer][i] = reward
                pop_bar.update(1)
            #print("Invalid rate: {:.2f}%({}/{})".format(invalid_count/num_pop*100, invalid_count, num_pop))
            bar_log.set_description_str(desc=f"Invalid rate: {invalid_count/num_pop*100:.2f}%({invalid_count}/{num_pop})")
            layer_bar.update(1)
            pop_bar.update(-num_pop)

        print("\n\n\n\n")
        # HW -> Mapping Opt.
        iteration = 0
        #hw_gen_bar = tqdm(total=num_gen, desc="GA search for HW", position=0)
        #hw_gen_log = tqdm(total=0, position=1, bar_format='{desc}')
        #map_gen_bar = tqdm(total=num_gen, position=2)
        #map_gen_log = tqdm(total=0, position=3, bar_format='{desc}')
        for hw_generation in range(num_gen):
            best_gen_reward =None
            parents = HWGene.select_parents(hw_new_population, hw_fitness,
                                            num_parents)

            offspring_crossover = HWGene.crossover(parents,
                                            num_pop-num_parents)
            offspring_mutation = HWGene.mutation(offspring_crossover)

            hw_new_population[0:parents.shape[0], :] = parents
            hw_new_population[parents.shape[0]:, :] = offspring_mutation
            num_invalid_node = 0

            for hw_pop_i in range(num_pop):
                hw_gene = hw_new_population[hw_pop_i] #Select HW Gene

                #Optimize Mapping Gene
                tot_hw_reward = 0.0
                total_used_constraint = 0.0
                for lr_i in range(num_layers):
                    map_iter = 0
                    #map_gen_bar.set_description_str(desc="GA search for mapping layer {}".format(lr_i))
                    self.map_best_reward[lr_i] = float("-Inf")
                    for map_generation in range(num_gen):
                        best_gen_map_reward = None
                        parents = MapGene.select_parents(map_new_population[lr_i], map_fitness[lr_i],
                                                        num_parents)
                        offspring_crossover = MapGene.crossover(parents,
                                                        num_pop-num_parents)
                        offspring_mutation = MapGene.mutation(offspring_crossover)

                        map_new_population[lr_i][0:parents.shape[0], :] = parents
                        map_new_population[lr_i][parents.shape[0]:, :] = offspring_mutation
                        map_num_invalid_node = 0
                        for map_pop_i in range(num_pop):
                            map_gene = map_new_population[lr_i][map_pop_i]
                            reward, used_constraint = self.exterior_search(self.model_defs[lr_i], hw_gene, map_gene)
                            if reward is None:
                                reward = float("-Inf")
                                #print("Error with reward")
                                #print(new_population[i])
                                #exit(-1)
                                map_num_invalid_node += 1
                            elif used_constraint > self.constraint_value:
                                reward = float("-Inf")
                                map_num_invalid_node += 1
                            if reward > self.map_best_reward[lr_i]:
                                best_gen_map_reward = reward
                                self.map_best_reward[lr_i] = reward
                                self.map_best_sol[lr_i] = map_gene
                            map_fitness[lr_i][map_pop_i] = reward
                        self.map_best_rewards_iteration[lr_i].append(map_iter)
                        self.map_best_rewards[lr_i].append(self.map_best_reward[lr_i])
                        map_iter += 1
                        #map_gen_bar.update(1)
                        if best_gen_map_reward  is not None:
                            #self.fd.write("\nMap Generation {}: new best award reward: {:9e}".format(map_generation+1, self.map_best_reward[lr_i])) if self.fd else None
                            print("\nMap Generation {} for HWgen{} Layer{}: new best award reward: {:9e}".format(map_generation+1, hw_generation, lr_i, self.map_best_reward[lr_i]))
                            #map_gen_log.set_description_str("Map Generation {}: new best award reward: {:9e}".format(map_generation+1, self.map_best_reward[lr_i]))
                    reward, used_constraint = self.exterior_search(self.model_defs[lr_i], hw_gene, self.map_best_sol[lr_i])
                    print("Observation ", self.observation)
                    #print(self.map_best_reward[lr_i])
                    #print(self.model_defs[lr_i], hw_gene, self.map_best_sol[lr_i])
                    print(reward, used_constraint)
                    if reward == None:
                        tot_hw_reward=None
                        total_used_constraint = None
                        print("None")
                        break
                    tot_hw_reward += reward
                    total_used_constraint += used_constraint
                    #map_gen_bar.update(-num_gen)
                reward = tot_hw_reward
                if reward is None:
                    reward = float("-Inf")
                    #print("Error with reward")
                    #print(new_population[i])
                    #exit(-1)
                elif total_used_constraint//num_layers > self.constraint_value:
                    reward = float("-Inf")
                    num_invalid_node += 1
                if reward > self.best_reward:
                    best_gen_reward = reward
                    self.best_reward = reward
                    self.best_reward_constraint = total_used_constraint/num_layers
                    self.best_sol = hw_gene

                hw_fitness[hw_pop_i] = reward
                iteration += 1
                self.best_rewards_iteration.append(iteration)
                self.best_rewards_constraint.append(self.best_reward_constraint)
                self.best_rewards.append(self.best_reward)
            if best_gen_reward  is not None:
                #self.fd.write("\nHW Generation {}: new best award reward: {:9e}".format(hw_generation+1, self.best_reward)) if self.fd else None
                print("\nHW Generation {}: new best award reward: {:9e}".format(hw_generation+1, self.best_reward))
                #hw_gen_log.set_description_str("HW Generation {}: new best award reward: {:9e}".format(hw_generation+1, self.best_reward))
            self.count_invalid += num_invalid_node
            self.save_chkpt()
            #hw_gen_bar.update(1)
        self.save_chkpt()
        print("Best result total:")
        print(self.best_rewards)
        print("Constraints of best")
        print(self.best_rewards_constraint)


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

    def exterior_search_special(self, actions, action_size=2):
        raise "Depleted"
        total_reward = None
        mac_rec_list = []
        latency_list = []
        total_constraint = 0
        for i in range(len(actions)):
            action = actions[i]
            maestro_state = np.concatenate((self.model_defs[i], action))
            reward, constraint = self.oberserve_maestro(maestro_state)
            if reward == None:
                return None
            else:
                mac_rec_list.append(self.observation[-1])
                latency_list.append(self.observation[0])
            total_constraint += constraint
        total_reward = sum(mac_rec_list)/sum(latency_list)
        return total_reward, total_constraint

    def exterior_search_all_layer(self, model, hw_gene, map_genes, num_pop): #layer_info = dimension
        tot_reward = 0
        tot_constraint = 0
        for i in range(len(model)):
            #reward, constraint = self.oberserve_maestro(model[i], hw_gene, map_genes[i][num_pop])
            reward, constraint = self.exterior_search(model[i], hw_gene, map_genes[i][num_pop])
            if reward == None:
                return None, None
            tot_reward += reward
            tot_constraint = constraint
        return tot_reward, tot_constraint
    def exterior_search(self, layer_info, hw_gene, map_gene):
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
            reward, constraint = self.oberserve_maestro(layer_info, hw_gene, map_gene)
            self.exp_table[table_entry] = (reward, constraint)
        if reward == None:
            return None, None
        total_reward = reward
        total_constraint = constraint
        #print("estiated: ", total_reward,total_constraint)
        return total_reward, total_constraint
        
    def write_maestro_dataflow(self, dimension, dataflow, m_file=None, layer_id=0):
        if len(dimension) > 6:
            m_type = m_type_dicts[int(dimension[-1])]
        else:
            m_type = "CONV"
        # Dataflow description
        with open("../../data/dataflow/dpt.m", "r") as fdpt:
            fo = open("{}.m".format(m_file), "w")
            fo.write("Network {} {{\n".format(layer_id))
            fo.write("Layer {} {{\n".format(m_type))
            fo.write("Type: {}\n".format(m_type))
            fo.write(
                "Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(
                    *dimension))
            if m_type == "CONV":
                fo.write("Dataflow {\n")
                for ln in range(dataflow.length()):
                    [df_type, df_var, df_size1, df_size2] = dataflow.read_line(ln)
                    if df_type == "Temporal" or df_type == "Spatial":
                        fo.write("{}Map({},{}) {};\n".format(df_type,df_size1,df_size2,df_var))
                    elif df_type == "Cluster":
                        fo.write("{}({},{});\n".format(df_type,df_size1,df_var))
                    else:
                        print(df_type)
                        raise "Exception Type"
                fo.write("}\n")
            else:
                fdpt.seek(0)
                fo.write(fdpt.read())
            fo.write("}\n")
            fo.write("}")
            fo.close()

    def softmax(self, a) :
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def gene2dataflow(self, dimension, hw_gene, map_gene):
        dataflow = MAESTRO_DATAFLOW()
        size_hw_dim_0 = int(hw_gene[HW_GENE.DIM_SIZE_0])
        size_hw_dim_1 = int(hw_gene[HW_GENE.DIM_SIZE_1])
        size_hw_dim_2 = int(hw_gene[HW_GENE.DIM_SIZE_2])
        selected_hw_dim = sorted(list(enumerate(hw_gene[HW_GENE.PAR_DIM_K:HW_GENE.PAR_DIM_S+1])), key=lambda x:x[1])[-int(hw_gene[HW_GENE.NUM_DIM]):]
        sorted_arr_map_dim = sorted(list(enumerate(map_gene[MAPPING_GENE.ARR_LOOP_ORDER_K:MAPPING_GENE.ARR_LOOP_ORDER_S+1])), key=lambda x:x[1])
        sorted_pe_map_dim = sorted(list(enumerate(map_gene[MAPPING_GENE.PE_LOOP_ORDER_K:MAPPING_GENE.PE_LOOP_ORDER_S+1])), key=lambda x:x[1])

        dim_size = [int(hw_gene[HW_GENE.NUM_PE]**x) for x in self.softmax([hw_gene[HW_GENE.DIM_SIZE_0], hw_gene[HW_GENE.DIM_SIZE_1]])] #get dim size from encoding vector
        tile_size = [int(x)+1 for x in np.array(map_gene[0:6]) * np.array(dimension[0:6])]
        #FIXME!
        tile_size[4] = dimension[4]
        #FIXME!
        tile_size[5] = dimension[5]
        #FIXME!
        in_tile_size = [1, 1, 1, 1, dimension[4], dimension[5]]
        #FIXME!
        #print("Dim size: ", dim_size, (hw_gene[HW_GENE.NUM_PE]))
        #print("Tile size: ", tile_size, map_gene[0:6], dimension[0:6], sorted_arr_map_dim)

        first_hw_dim = [idx for idx, item in sorted_arr_map_dim].index(selected_hw_dim[0][0]) #First hw dim
        for i in range(6):
            if i is not first_hw_dim: dataflow.push_temporal_outer(sorted_arr_map_dim[i][0], tile_size[sorted_arr_map_dim[i][0]])
        #dataflow.push_spatial_outer(selected_hw_dim[0][0], hw_gene[HW_GENE.DIM_SIZE_0])
        dataflow.push_spatial_inner(selected_hw_dim[0][0])
        #dataflow.push_cluster(dim_size[0]) #dim 0 is implied by #PE/dim1
        dataflow.push_cluster(dim_size[1])
        second_hw_dim = [idx for idx, item in sorted_pe_map_dim].index(selected_hw_dim[1][0]) #First hw dim
        for i in range(6):
            if i is not second_hw_dim: dataflow.push_temporal_outer(sorted_pe_map_dim[i][0], in_tile_size[sorted_pe_map_dim[i][0]])
            #if i is not second_hw_dim: dataflow.push_temporal_inner(sorted_pe_map_dim[i][0])
        dataflow.push_spatial_inner(selected_hw_dim[1][0])
        #dataflow.push_cluster(dim_size[1])
        return dataflow
        
    #def oberserve_maestro(self, state, firsttime=False):
    def oberserve_maestro(self, dimension, hw_gene, map_gene, firsttime=False, multi=False):
        if multi==True:
            if (len(dimension) != len(hw_gene)) or (len(dimension) != len(map_gene)):
                raise "Invalid Argument"
            lim = len(dimension)
        else:
            lim = 1
            dimension = [dimension]
            hw_gene = [hw_gene]
            map_gene = [map_gene]
        process = []
        main_m_file = self.random_file_name
        for i in range(lim):
            m_file = main_m_file+"{}".format(i)
            self.write_maestro_dataflow (dimension[i], self.gene2dataflow(dimension[i], hw_gene[i], map_gene[i]), m_file=m_file)

            #HW gene
            num_pes = hw_gene[i][2]
            l1_size_cstr = hw_gene[i][1]
            l2_size_cstr = hw_gene[i][0]
            noc_bw_cstr = hw_gene[i][3]
            dim_size = [int(hw_gene[i][HW_GENE.NUM_PE]**x) for x in self.softmax([hw_gene[i][HW_GENE.DIM_SIZE_0], hw_gene[i][HW_GENE.DIM_SIZE_1]])] #get dim size from encoding vector)
            os.remove("./{}.csv".format(m_file)) if os.path.exists("./{}.csv".format(m_file)) else None
            command = [self._executable,
                    "--Mapping_file={}.m".format(m_file),
                    "--full_buffer=false",
                    "--noc_bw_cstr={}".format(int(noc_bw_cstr)),
                    "--noc_hops=1",
                    "--noc_hop_latency=1",
                    "--noc_mc_support=true",
                    #"--num_pes={}".format(int(num_pes)),
                    "--num_pes={}".format(int(dim_size[0]*dim_size[1])),
                    #"--num_simd_lanes=1",
                    "--l1_size_cstr={}".format(int(l1_size_cstr)),
                    "--l2_size_cstr={}".format(int(l2_size_cstr)),
                    "--print_res=false", "--print_res_csv_file=true", "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]

            process.append(Popen(command, stdout=PIPE, stderr=PIPE))
        for i in range(lim):
            stdout, stderr = process[i].communicate()
            process[i].wait()

        #print(command, stdout, self.gene2dataflow(dimension, hw_gene, map_gene).mapping)
        try:
            df = pd.read_csv("./{}.csv".format(m_file))
            layer_name = df[" Layer Number"]
            runtime = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
            area = np.array(df[" Area"]).reshape(-1, 1)
            power = np.array(df[" Power"]).reshape(-1, 1)
            l1_size = np.array(df[" L1 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            l2_size = np.array(df["  L2 SRAM Size Req (Bytes)"]).reshape(-1, 1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            os.remove("./{}.csv".format(m_file))  if os.path.exists("./{}.csv".format(m_file)) else None
            os.remove("./log.txt") if os.path.exists("./log.txt") else None
            #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            #    print(df)
            #print(hw_gene[0], map_gene[0], stdout)
            #print(type(int(runtime[0][0])))
            if runtime[0][0] <= 0 or energy[0][0] <= 0 or throughput[0][0] <= 0:
                #print(runtime[0][0])
                #print(energy[0][0])
                #print(throughput[0][0])
                raise
            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power]]
            #print("pass")
            return self.judge()
        except:
            # Compile Error -> Constraint(panelty)
            #print("+"*20)
            #print(num_pe, KTileSz, ClusterSz)
            #print("+" * 20)
            return None, None


    def write_maestro_gemm(self, dimension, dataflow="dla", KTileSz=1, CTileSz=1, ClusterSz=4, m_file=None, layer_id=0,df_idx=None):
        raise "Depleted"
        if df_idx is not None:
            dataflow = df_dict[df_idx]
        m_type = "CONV"
        SzM, SzN, SzK = dimension
        dimension = [SzN, SzK, SzM, 1, 1, 1]
        with open("{}_f.m".format(dataflow), "r") as fd:
            with open("dpt_f.m", "r") as fdpt:
                with open("{}.m".format(m_file), "w") as fo:
                    fo.write("Constant KTileSz {};\n".format(KTileSz))
                    fo.write("Constant CTileSz {};\n".format(CTileSz))
                    fo.write("Constant ClusterSz {};\n".format(ClusterSz))
                    fo.write("Network {} {{\n".format(layer_id))
                    fo.write("Layer {} {{\n".format("CONV"))
                    fo.write("Type: {}\n".format(m_type))
                    fo.write(
                        "Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(
                            *dimension))
                    if m_type == "CONV":
                        fd.seek(0)
                        fo.write(fd.read())
                    else:
                        fdpt.seek(0)
                        fo.write(fdpt.read())
                    fo.write("}\n")
                    fo.write("}")
