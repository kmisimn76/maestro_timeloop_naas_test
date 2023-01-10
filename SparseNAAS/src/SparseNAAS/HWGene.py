
import random
import numpy as np

from enum import IntEnum
class HW_GENE (IntEnum):
    L2_SIZE     = 0 #deprecated
    L1_SIZE     = 1 #deprecated
    NUM_PE      = 2
    BW          = 3 #deprecated
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
    GROUP_DENSITY= 14
    BANK        = 15
    L2_WEIGHT_SIZE     = 16
    L2_INPUT_SIZE     = 17
    L2_OUTPUT_SIZE     = 18
    L1_WEIGHT_SIZE     = 19
    L1_INPUT_SIZE     = 20
    L1_OUTPUT_SIZE     = 21
    USE_SPARSITY = 22


## Design Space Constraints
'''
#Embedded
MAX_PE = 8#12#14 #12 #8
MIN_PE = 7#10#12 #11 #8
SQUARE_SHAPE_PE = False
WEIGHT_STATIONARY = False
DISABLE_BANK = False
DISABLE_GROUP = False #sparse/dense
MIN_BANK = 1#0#1 #sparse/dense
MAX_GROUP = 4#6#4 #sparse/dense
MAX_L2_SIZE = 64000#4*8*128000
'''

MAX_PE = 10#12#14 #12 #8 #Exponential of 2
MIN_PE = 8#10#12 #11 #8 #Exponential of 2
SQUARE_SHAPE_PE = False
WEIGHT_STATIONARY = False
OUTPUT_STATIONARY = True#False
USE_SPARSITY = 0#0 #0=random, 1=dense, 2=sparse

DISABLE_BANK = False
DISABLE_GROUP = False #sparse/dense
MIN_BANK = 0#1 #sparse/dense #Exponential of 2
MAX_GROUP = 6#3#6#4 #sparse/dense #Exponential of 2
MAX_L2_SIZE = 8*128000#4*8*128000 # # of Elements
specific_gene = None
#specific_gene = [3000, 100, 10, 50, 2, 0.6, 0.4, 0.5, 5,6,4,3,2,1, 4, 1, 310167.0,490008.0,460822.0,170699.70811429684/310167.0,37488.6041682779/490008.0,59665.545348994056/460822.0, 1]
#specific_gene = [3000, 100, 10, 50, 2, 0.6, 0.4, 0.5, 5,6,4,3,2,1, 4, 1, 310167.0/2,490008.0,460822.0,170699.70811429684/310167.0,37488.6041682779/490008.0,59665.545348994056/460822.0, 1]
#specific_gene = [3000, 100, 10, 50, 2, 0.6, 0.4, 0.5, 5,4,3,6,2,1, 4, 2, 384726.0,180530.0,80689.0,77220.32072257111/384726.0,129800.6326891591/180530.0,54683.2319325427/80689.0, 0]
#specific_gene = [3000, 100, 10, 50, 2, 0.5, 0.5, 0.5, 5,4,3,6,2,1, 2, 1, 219928.0,183399.0,150548.0,33623.339297894214/219928.0,135077.02740795567/183399.0,126948.65197363592/150548.0, 1]



class _HWGene(object):
    def get_sample_gene(self):
        return [3000, 100, 8, 50, 2, 0.5, 0.5, 0.5, 6,5,4,3,2,1, 1, 4, 10000, 10000, 10000, 0.5, 0.5, 0.5, 1]
    def generate_random_gene(self):
        if specific_gene is not None:
            return [g for g in specific_gene]
        L2Buf = random.random()
        L1Buf = random.random()
        L2Buf_weight = random.randint(100, MAX_L2_SIZE) # # of Elements
        L2Buf_input = random.randint(100, MAX_L2_SIZE)
        L2Buf_output = random.randint(100, MAX_L2_SIZE)
        L1Buf_weight = random.random()
        L1Buf_input = random.random()
        L1Buf_output = random.random()
        PEs = random.randint(MIN_PE, MAX_PE) # power of 2 #300
        BW = random.randint(1,1000)
        NumDim = random.randint(2,2) #FIXME : Hardcoding, Fix NumDim to 2
        #NumDim = random.randint(2,3) #FIXME : Hardcoding, Fix NumDim to 2

        if SQUARE_SHAPE_PE:
            DimSize0 = 0.5 #for dense FIXME 0~1 -> to softmax
            DimSize1 = 0.5 #for dense FIXME 0~1
            DimSize2 = 0.5 #for dense FIXME 0~1
        else:
            DimSize0 = random.random() #0~1 -> to softmax
            DimSize1 = random.random() #0~1
            DimSize2 = random.random() #0~1

        if WEIGHT_STATIONARY:
            ParDimK = 6#random.randint(1,6) #for dense
            ParDimC = 5#random.randint(1,6) #for dense
            ParDimY = 4#random.randint(1,6) #for dense
            ParDimX = 3#random.randint(1,6) #for dense
            ParDimR = 2#random.randint(1,6) #for dense
            ParDimS = 1#random.randint(1,6) #for dense
        elif OUTPUT_STATIONARY:
            ParDimK = 5#random.randint(1,6) #for dense
            ParDimX = 6#random.randint(1,6) #for dense
            ParDimY = 4#random.randint(1,6) #for dense
            ParDimC = 3#random.randint(1,6) #for dense
            ParDimR = 2#random.randint(1,6) #for dense
            ParDimS = 1#random.randint(1,6) #for dense
        else:
            ParDimK = random.randint(1,6)
            ParDimC = random.randint(1,6)
            #ParDimC = random.randint(1,1)
            ParDimY = random.randint(1,6)
            ParDimX = random.randint(1,6)
            #ParDimR = random.randint(1,6)
            #ParDimS = random.randint(1,6)
            ParDimR = random.randint(1,1)
            ParDimS = random.randint(1,1)

        if DISABLE_BANK:
            Bank    = 0 # for Dense FIXME
        else:
            #Bank    = random.randint(0, 4) # power of 2
            Bank    = random.randint(MIN_BANK, 4) # power of 2
        if DISABLE_GROUP:
            Density = 5 # for Dense FIXME
        else:
            #Density = random.randint(0,6) # power of 2
            Density = random.randint(0,MAX_GROUP) # power of 2

        if USE_SPARSITY == 0:
            use_sparsity = random.randint(0, 1)
        elif USE_SPARSITY == 1:
            use_sparsity = 0
        else:
            use_sparsity = 1

        return [L2Buf, L1Buf, PEs, BW, NumDim,
                    DimSize0, DimSize1, DimSize2,
                    ParDimK, ParDimC, ParDimY, ParDimX, ParDimR, ParDimS, Density, Bank,
                    L2Buf_weight, L2Buf_input, L2Buf_output, L1Buf_weight, L1Buf_input, L1Buf_output, use_sparsity]

    #HWGene
    def select_parents(self, pop, fitness, num_parents, gen=0):
        for i in range(len(pop)):
            for j in range(i+1, len(pop)):
                if  pop[i][2] == pop[j][2] and \
                    pop[i][4] == pop[j][4] and \
                    pop[i][5] == pop[j][5] and \
                    pop[i][6] == pop[j][6] and \
                    pop[i][7] == pop[j][7] and \
                    pop[i][8] == pop[j][8] and \
                    pop[i][9] == pop[j][9] and \
                    pop[i][10] == pop[j][10] and \
                    pop[i][11] == pop[j][11] and \
                    pop[i][12] == pop[j][12] and \
                    pop[i][13] == pop[j][13] and \
                    pop[i][14] == pop[j][14] and \
                    pop[i][15] == pop[j][15] and \
                    pop[i][16] == pop[j][16] and \
                    pop[i][17] == pop[j][17] and \
                    pop[i][18] == pop[j][18] and \
                    pop[i][19] == pop[j][19] and \
                    pop[i][20] == pop[j][20] and \
                    pop[i][21] == pop[j][21] : #중복 gene 제거
                    fitness[j] = float('-inf')
        for i in range(len(pop)): # search more sparser
            dead = False
            if pop[i][HW_GENE.GROUP_DENSITY] > (6 - gen*((6-3)/200)): # gen: GRUOP const 6 -> 3
                dead = True
            if gen > 100:
                if pop[i][HW_GENE.USE_SPARSITY] == False and random.random() <= (gen/200):
                    dead = True
            if dead==True and USE_SPARSITY!=1:
                fitness[i] = float('-inf')

        parents = np.empty((num_parents, len(HW_GENE)))
        sortarg = np.array(fitness).argsort()
        for parent_num in range(num_parents//2):
            #max_fitness_idx = np.where(fitness == np.max(fitness))
            #max_fitness_idx = max_fitness_idx[0][0]
            #parents[parent_num] = pop[max_fitness_idx]
            #fitness[max_fitness_idx] = float("-inf")
            parents[parent_num] = pop[sortarg[len(pop)-1-parent_num]]
        for parent_num in range(num_parents//2, num_parents):
            parents[parent_num] = pop[sortarg[random.randint(0,len(pop)-1-num_parents//2)]]
        return parents

    #HWGene
    def crossover(self, parents, offspring_size):
        offspring = np.empty((offspring_size, len(HW_GENE)))
        for k in range(offspring_size):
            parent1_idx = k%parents.shape[0]
            parent2_idx = np.random.randint(0, parents.shape[0])
            for i in range(len(HW_GENE)):
                offspring[k][i] = parents[parent1_idx][i] if random.randint(0,1)==0 else parents[parent2_idx][i] #crossover 1:1
                #offspring[k][i] = parents[parent1_idx][i] if 0==0 else parents[parent2_idx][i] #crossover 1:1
        return offspring

    #HWGene
    def mutation(self, offsprings,rate=0.05):
        for idx in range(offsprings.shape[0]):
            rand_list = self.generate_random_gene()
            for p in range(offsprings.shape[1]):
                if random.random() < rate:
                    offsprings[idx][p] = rand_list[p]
        return offsprings

class DIM (IntEnum):
    K = 0
    C = 1
    H = 2
    W = 3
    R = 4
    S = 5
    P = 6
    def to_str(data):
        if data==DIM.K: return "K"
        if data==DIM.C: return "C"
        if data==DIM.H: return "H"
        if data==DIM.W: return "W"
        if data==DIM.R: return "R"
        if data==DIM.S: return "S"
        if data==DIM.P: return "P"
        if data==None: return "None"
        #print(data)
        raise "Exception"
    def to_str_timeloop(data):
        if data==DIM.K: return "M"
        if data==DIM.C: return "C"
        if data==DIM.H: return "P"
        if data==DIM.W: return "Q"
        if data==DIM.R: return "R"
        if data==DIM.S: return "S"
        if data==DIM.P: return "Z"
        if data==None: return "None"
        #print(data)
        raise "Exception"

class TIMELOOP_HW:
    def softmax(self, a) :
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def __init__(self):
        self.X = 0
        self.Y = 0
        self.XDim = DIM.W
        self.YDim = DIM.H
        self.L2_size = 0
        self.L1_size = 0
        self.Density = 0
        self.Bank = 0
        self.L2_weight_size = 0
        self.L2_input_size = 0
        self.L2_output_size = 0
        self.L1_weight_size = 0
        self.L1_input_size = 0
        self.L1_output_size = 0
        self.use_sparsity = True
    def set_HW(self,hw_gene):
        num_dim = int(hw_gene[HW_GENE.NUM_DIM])
        selected_hw_dim = sorted(list(enumerate(hw_gene[HW_GENE.PAR_DIM_K:HW_GENE.PAR_DIM_S+1])), key=lambda x:x[1])[-num_dim:]
        #sum_dim = hw_gene[HW_GENE.DIM_SIZE_0] + hw_gene[HW_GENE.DIM_SIZE_1]
        #dim_size = [2**(round(hw_gene[HW_GENE.NUM_PE]*(x/sum_dim))) for x in [hw_gene[HW_GENE.DIM_SIZE_0], hw_gene[HW_GENE.DIM_SIZE_1]]] #get dim size from encoding vector
        sum_dim = sum(hw_gene[HW_GENE.DIM_SIZE_0:HW_GENE.DIM_SIZE_0+num_dim])
        dim_size = [2**(round(hw_gene[HW_GENE.NUM_PE]*(x/sum_dim))) for x in [hw_gene[HW_GENE.DIM_SIZE_0 + i] for i in range(num_dim)]] #get dim size from encoding vector
        x = dim_size[0]
        y = dim_size[1] if num_dim >= 2 else None
        z = dim_size[2] if num_dim >= 3 else None
        xdim = selected_hw_dim[0][0]
        ydim = selected_hw_dim[1][0] if num_dim >= 2 else None
        zdim = selected_hw_dim[2][0] if num_dim >= 3 else None
        l2_size = hw_gene[HW_GENE.L2_SIZE]
        l1_size = hw_gene[HW_GENE.L1_SIZE]
        density = 2**(int(hw_gene[HW_GENE.GROUP_DENSITY] ))
        bank = 2**int(hw_gene[HW_GENE.BANK])
        self.num_dim = num_dim
        self.dim_size = dim_size
        self.X = x
        self.Y = y
        self.Z = z
        self.XDim = xdim
        self.YDim = ydim
        self.ZDim = zdim
        self.L2_size = l2_size
        self.L1_size = l1_size
        self.Density = density
        self.Bank = bank
        self.use_sparsity = True if hw_gene[HW_GENE.USE_SPARSITY] == 1 else False
        #buffer_scale_for_sparse = 0.5 if self.use_sparsity is True else 1.0 # for smooth GA
        buffer_scale_for_sparse = 1.0

        self.L2_weight_size = hw_gene[HW_GENE.L2_WEIGHT_SIZE] * buffer_scale_for_sparse
        self.L2_input_size = hw_gene[HW_GENE.L2_INPUT_SIZE] * buffer_scale_for_sparse
        self.L2_output_size = hw_gene[HW_GENE.L2_OUTPUT_SIZE] * buffer_scale_for_sparse
        self.L1_weight_size = hw_gene[HW_GENE.L1_WEIGHT_SIZE] * hw_gene[HW_GENE.L2_WEIGHT_SIZE]
        self.L1_input_size = hw_gene[HW_GENE.L1_INPUT_SIZE] * hw_gene[HW_GENE.L2_INPUT_SIZE]
        self.L1_output_size = hw_gene[HW_GENE.L1_OUTPUT_SIZE] * hw_gene[HW_GENE.L2_OUTPUT_SIZE]
    def get_num_dim(self):
        return self.num_dim
    def get_X(self):
        return self.X
    def get_Y(self):
        return self.Y
    def get_Z(self):
        return self.Z
    def get_XDim(self):
        return self.XDim
    def get_YDim(self):
        return self.YDim
    def get_ZDim(self):
        return self.ZDim
    def get_L2_size(self):
        return self.L2_weight_size, self.L2_input_size, self.L2_output_size
    def get_L1_size(self):
        return self.L1_weight_size, self.L1_input_size, self.L1_output_size
    def get_group_density(self):
        #density = self.Density if self.Density <= 2**4 else (self.X if (self.XDim==2 or self.XDim==3) else (self.Y if (self.YDim==2 or self.YDim==3) else (self.X if (self.XDim==1) else (self.Y if (self.YDim==1) else 1) ) ) )
        DIM = (self.X if (self.XDim==2 or self.XDim==3) else (self.Y if (self.YDim==2 or self.YDim==3) else (self.X if (self.XDim==1) else (self.Y if ( self.YDim==1) else 1)))) #2=H, 3=W, 1=C
        density = self.Density if self.Density <= DIM else DIM
        if self.use_sparsity is False:
            density = DIM
        return density
    def get_bank(self):
        return self.Bank
    def get_use_sparsity(self):
        return self.use_sparsity

