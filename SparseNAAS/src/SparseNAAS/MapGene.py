import random
import numpy as np

from enum import IntEnum

from HWGene import *

class MAPPING_GENE (IntEnum):
    ARR_LOOP_ORDER_K = 0
    ARR_LOOP_ORDER_C = 1
    ARR_LOOP_ORDER_Y = 2
    ARR_LOOP_ORDER_X = 3
    ARR_LOOP_ORDER_R = 4
    ARR_LOOP_ORDER_S = 5
    ARR_TILE_SIZE_K  = 6
    ARR_TILE_SIZE_C  = 7
    ARR_TILE_SIZE_Y  = 8
    ARR_TILE_SIZE_X  = 9
    ARR_TILE_SIZE_R  = 10
    ARR_TILE_SIZE_S  = 11
    PE_LOOP_ORDER_K  = 12
    PE_LOOP_ORDER_C  = 13
    PE_LOOP_ORDER_Y  = 14
    PE_LOOP_ORDER_X  = 15
    PE_LOOP_ORDER_R  = 16
    PE_LOOP_ORDER_S  = 17
    L1_TILE_SIZE_K  = 18
    L1_TILE_SIZE_C  = 19
    L1_TILE_SIZE_Y  = 20
    L1_TILE_SIZE_X  = 21
    L1_TILE_SIZE_R  = 22
    L1_TILE_SIZE_S  = 23

## Design Space Constraints

class _MapGene(object):
    def get_sample_gene(self, seed1=None, seed2=None):
        #return [6,5,4,3,2,1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 6,5,4,3,2,1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        rd1 = random.random()*0.8+0.1 if seed1 is None else seed1
        rd2 = random.random()*0.8+0.1 if seed2 is None else seed2
        return [6,5,4,3,2,1, rd1,rd1,rd1,rd1,rd1,rd1, 6,5,4,3,2,1, rd2,rd2,rd2,rd2,rd2,rd2]
        #return [6,5,4,3,2,1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 6,5,4,3,2,1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    def generate_random_gene(self):
        ArrTileSizeK = random.random() #0~1
        ArrTileSizeC = random.random()
        ArrTileSizeY = random.random()
        ArrTileSizeX = random.random()
        ArrTileSizeR = random.random()
        ArrTileSizeS = random.random()
        L1TileSizeK = random.random() #0~1
        L1TileSizeC = random.random()
        L1TileSizeY = random.random()
        L1TileSizeX = random.random()
        L1TileSizeR = random.random()
        L1TileSizeS = random.random()
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
                    ArrLoopOrderK,ArrLoopOrderC,ArrLoopOrderY,ArrLoopOrderX,ArrLoopOrderR,ArrLoopOrderS,
                    ArrTileSizeK,ArrTileSizeC,ArrTileSizeY,ArrTileSizeX,ArrTileSizeR,ArrTileSizeS,
                    PELoopOrderK,PELoopOrderC,PELoopOrderY,PELoopOrderX,PELoopOrderR,PELoopOrderS,
                    L1TileSizeK,L1TileSizeC,L1TileSizeY,L1TileSizeX,L1TileSizeR,L1TileSizeS
                ]

    #MapGene
    def select_parents(self, pop, fitness, num_parents):
        '''
        parents = np.empty((num_parents, len(MAPPING_GENE)))
        for parent_num in range(num_parents):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num] = pop[max_fitness_idx]
            fitness[max_fitness_idx] = float("-inf")
        '''
        for i in range(len(pop)):
            for j in range(i+1, len(pop)):
                if sum([(0 if pop[i][l] == pop[j][l] else 1) for l in range(len(MAPPING_GENE))]) == 0:
                    fitness[j] = float('-inf')

        parents = np.empty((num_parents, len(MAPPING_GENE)))
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

    #MapGene
    def crossover(self, parents, offspring_size):
        offspring = np.empty((offspring_size, len(MAPPING_GENE)))
        for k in range(offspring_size):
            parent1_idx = k%parents.shape[0]
            parent2_idx = np.random.randint(0, parents.shape[0])
            for i in range(len(MAPPING_GENE)):
                offspring[k][i] = parents[parent1_idx][i] if random.randint(0,1)==0 else parents[parent2_idx][i] #crossover 1:1
                #offspring[k][i] = parents[parent1_idx][i] if 0==0 else parents[parent2_idx][i] #crossover 1:1
        return offspring

    #MapGene
    def mutation(self, offsprings,rate=0.05):
        for idx in range(offsprings.shape[0]):
            rand_list = self.generate_random_gene()
            for p in range(offsprings.shape[1]):
                if random.random() < rate:
                    offsprings[idx][p] = rand_list[p]
        return offsprings

import math
class TIMELOOP_MAPPING:
    def __init__(self):
        self.mapping_gene_raw = None
        self.l_info = None # layer information : K C Y X R S
        self.mapping_tile_size = None
        self.mapping_array_order = None
        self.mapping_pe_order = None
    def set_mapping_gene(self, l_info, dim_sz, hw_gene, hw_info, gene):
        self.mapping_gene_raw = gene
        self.l_info = l_info
        # order
        self.dim_size = dim_sz
        selected_hw_dim = sorted(list(enumerate(hw_gene[HW_GENE.PAR_DIM_K:HW_GENE.PAR_DIM_S+1])), key=lambda x:x[1])[-int(hw_gene[HW_GENE.NUM_DIM]):]
        sorted_arr_map_dim = sorted(list(enumerate(self.mapping_gene_raw[MAPPING_GENE.ARR_LOOP_ORDER_K:MAPPING_GENE.ARR_LOOP_ORDER_S+1])), key=lambda x:x[1])
        sorted_pe_map_dim = sorted(list(enumerate(self.mapping_gene_raw[MAPPING_GENE.PE_LOOP_ORDER_K:MAPPING_GENE.PE_LOOP_ORDER_S+1])), key=lambda x:x[1])
        self.mapping_array_order = [sorted_arr_map_dim[i][0] for i in range(0,6)]
        self.mapping_pe_order = [sorted_pe_map_dim[i][0] for i in range(0,6)]
        self.mapping_selected_hw_dim = [selected_hw_dim[0][0], selected_hw_dim[1][0]]

        # tile size
        self.l_info_hw = [v for v in l_info]
        self.l_info_hw[self.mapping_selected_hw_dim[0]] = int(math.ceil(self.l_info_hw[self.mapping_selected_hw_dim[0]]/self.dim_size[0]))
        self.l_info_hw[self.mapping_selected_hw_dim[1]] = int(math.ceil(self.l_info_hw[self.mapping_selected_hw_dim[1]]/self.dim_size[1]))
        self.mapping_tile_size = [int(math.ceil(self.mapping_gene_raw[i]*self.l_info_hw[j])) for (i, j) in zip(range(MAPPING_GENE.ARR_TILE_SIZE_K, MAPPING_GENE.ARR_TILE_SIZE_S+1), range(0,6))]
        self.mapping_inner_tile_size = [int(math.ceil(self.mapping_gene_raw[i]*self.mapping_tile_size[j])) for (i, j) in zip(range(MAPPING_GENE.L1_TILE_SIZE_K, MAPPING_GENE.L1_TILE_SIZE_S+1), range(0,6))]
        self.mapping_inner_tile_size[self.mapping_selected_hw_dim[0]] = 1
        self.mapping_inner_tile_size[self.mapping_selected_hw_dim[1]] = 1
        self.mapping_inner_tile_size[4] = 1 #R,S PE tile is 1. except
        self.mapping_inner_tile_size[5] = 1 #R,S PE tile is 1. except
        
        X = hw_info.get_X()
        XDim = hw_info.get_XDim()
        Y = hw_info.get_Y()
        YDim = hw_info.get_YDim()
        Z = hw_info.get_Z()
        ZDim = hw_info.get_ZDim()
        HW_dim = [1, 1, 1, 1, 1, 1] #K,C,H,W,R,S
        HW_dim[XDim] = X
        if YDim is not None:
            HW_dim[YDim] = Y
        if ZDim is not None:
            HW_dim[ZDim] = Z
        bank_ = hw_info.get_bank()
        # PE tile is multiple of Bank
        if HW_dim[2] == 1 and HW_dim[3]==1: #weight stationary
            self.mapping_inner_tile_size[3] = bank_ * math.ceil(self.mapping_inner_tile_size[3]/bank_) #W bank
        elif HW_dim[1] == 1: #output stationary
            self.mapping_inner_tile_size[1] = bank_ * math.ceil(self.mapping_inner_tile_size[1]/bank_) #C bank
        elif HW_dim[0] == 1: #input stationary
            self.mapping_inner_tile_size[0] = bank_ * math.ceil(self.mapping_inner_tile_size[0]/bank_) #K bank
        else:
            raise Exception("err")

        self.l2_tile_size = [int(math.ceil(self.l_info_hw[i]/self.mapping_tile_size[i])) for i in range(0,6)]
        self.l1_tile_size = [int(math.ceil(self.mapping_tile_size[i]/self.mapping_inner_tile_size[i])) for i in range(0,6)]
        self.pe_tile_size = [int(math.ceil(self.mapping_inner_tile_size[i])) for i in range(0,6)]
    def get_mapping_L2_tile_size(self):
        output = [v for v in self.l2_tile_size]
        return output
    def get_mapping_L1_tile_size(self):
        output = [v for v in self.l1_tile_size]
        return output
    def get_mapping_PE_tile_size(self):
        output = [v for v in self.pe_tile_size]
        return output
    def get_mapping_parallel_size(self):
        output = [1 for i in range(0,6)]
        output[self.mapping_selected_hw_dim[0]] = int(self.dim_size[0])
        output[self.mapping_selected_hw_dim[1]] = int(self.dim_size[1])
        #output[self.mapping_selected_hw_dim[2]] = int(self.dim_size[2])
        return output
    def get_mapping_l2_order(self):
        return [self.mapping_array_order[i] for i in range(0,6)]
    def get_mapping_l1_order(self):
        return [self.mapping_pe_order[i] for i in range(0,6)]

