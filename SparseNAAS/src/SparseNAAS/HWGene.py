
import random
import numpy as np
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

