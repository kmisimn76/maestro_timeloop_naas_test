import random
import numpy as np
class _MapGene(object):
    def get_sample_gene(self):
        return [6,5,4,3,2,1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 6,5,4,3,2,1]
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
                    ArrLoopOrderK,ArrLoopOrderC,ArrLoopOrderY,ArrLoopOrderX,ArrLoopOrderR,ArrLoopOrderS,
                    ArrTileSizeK,ArrTileSizeC,ArrTileSizeY,ArrTileSizeX,ArrTileSizeR,ArrTileSizeS,
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

class _MapGene(object):
    def get_sample_gene(self):
        return [6,5,4,3,2,1, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 6,5,4,3,2,1]
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
                    ArrLoopOrderK,ArrLoopOrderC,ArrLoopOrderY,ArrLoopOrderX,ArrLoopOrderR,ArrLoopOrderS,
                    ArrTileSizeK,ArrTileSizeC,ArrTileSizeY,ArrTileSizeX,ArrTileSizeR,ArrTileSizeS,
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

