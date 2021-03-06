
import random
import numpy as np

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
    GROUP_DENSITY= 14
    BANK        = 15

class _HWGene(object):
    def get_sample_gene(self):
        return [3000, 100, 8, 50, 2, 0.5, 0.5, 0.5, 6,5,4,3,2,1, 1, 4]
    def generate_random_gene(self):
        L2Buf = random.randint(1, 1000)
        L1Buf = random.randint(1, 1000)
        PEs = random.randint(1, 8) #300
        BW = random.randint(1,1000)
        NumDim = random.randint(2,2) #FIXME : Hardcoding, Fix NumDim to 2
        DimSize0 = random.random() #0~1 -> to softmax
        DimSize1 = random.random() #0~1
        DimSize2 = random.random() #0~1
        ParDimK = random.randint(1,6)
        ParDimC = random.randint(1,6)
        ParDimY = random.randint(1,6)
        ParDimX = random.randint(1,6)
        ParDimR = random.randint(1,6)
        ParDimS = random.randint(1,6)
        Density = random.randint(1,3) # power of 2
        Bank    = random.randint(1,4) # power of 2
        return [L2Buf, L1Buf, PEs, BW, NumDim,
                    DimSize0, DimSize1, DimSize2,
                    ParDimK, ParDimC, ParDimY, ParDimX, ParDimR, ParDimS, Density, Bank]

    #HWGene
    def select_parents(self, pop, fitness, num_parents):
        parents = np.empty((num_parents, len(HW_GENE)))
        for parent_num in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num] = pop[max_fitness_idx]
            fitness[max_fitness_idx] = float("-inf")
        return parents

    #HWGene
    def crossover(self, parents, offspring_size):
        offspring = np.empty((offspring_size, len(HW_GENE)))
        for k in range(offspring_size):
            parent1_idx = k%parents.shape[0]
            parent2_idx = np.random.randint(0, parents.shape[0])
            for i in range(len(HW_GENE)):
                offspring[k][i] = parents[parent1_idx][i] if random.randint(0,1)==0 else parents[parent2_idx][i] #crossover 1:1
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
        #print(data)
        raise "Exception"
    def to_str_timeloop(data):
        if data==DIM.K: return "M"
        if data==DIM.C: return "C"
        if data==DIM.Y: return "Q"
        if data==DIM.X: return "P"
        if data==DIM.R: return "R"
        if data==DIM.S: return "S"
        if data==DIM.P: return "Z"
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
        self.XDim = DIM.X
        self.YDim = DIM.Y
        self.L2_size = 0
        self.L1_size = 0
        self.Density = 0
        self.Bank = 0
    def set_HW(self,hw_gene):
        selected_hw_dim = sorted(list(enumerate(hw_gene[HW_GENE.PAR_DIM_K:HW_GENE.PAR_DIM_S+1])), key=lambda x:x[1])[-int(hw_gene[HW_GENE.NUM_DIM]):]
        sum_dim = hw_gene[HW_GENE.DIM_SIZE_0] + hw_gene[HW_GENE.DIM_SIZE_1]
        dim_size = [2**(round(hw_gene[HW_GENE.NUM_PE]*(x/sum_dim))) for x in [hw_gene[HW_GENE.DIM_SIZE_0], hw_gene[HW_GENE.DIM_SIZE_1]]] #get dim size from encoding vector
        x = dim_size[0]
        y = dim_size[1]
        xdim = selected_hw_dim[0][0]
        ydim = selected_hw_dim[1][0]
        l2_size = hw_gene[HW_GENE.L2_SIZE]
        l1_size = hw_gene[HW_GENE.L1_SIZE]
        density = 2**(int(hw_gene[HW_GENE.GROUP_DENSITY] ))
        bank = 2**int(hw_gene[HW_GENE.BANK])
        self.dim_size = dim_size
        self.X = x
        self.Y = y
        self.XDim = xdim
        self.YDim = ydim
        self.L2_size = l2_size
        self.L1_size = l1_size
        self.Density = density
        self.Bank = bank
    def get_X(self):
        return self.X
    def get_Y(self):
        return self.Y
    def get_XDim(self):
        return self.XDim
    def get_YDim(self):
        return self.YDim
    def get_L2_size(self):
        return self.L2_size
    def get_L1_size(self):
        return self.L1_size
    def get_group_density(self):
        return self.Density
    def get_bank(self):
        return self.Bank

