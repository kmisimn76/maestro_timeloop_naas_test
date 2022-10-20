import numpy as np

from HWGene import *
from MapGene import *

def trunc_and_strength_quant(data, mini, maxi):
    data = max(min(data, 1), 0) #trunc
    data = data*(maxi-mini) + mini
    data = round(data)
    return data

def inverse_trunc_and_strength_quant(data, mini, maxi):
    lamb = 1e-6
    data = (data - mini) / (maxi-mini+lamb)
    return data

def HWGene_mapper(vector):
    hw_gene = [0 for i in range(len(HW_GENE))]
    hw_gene[HW_GENE.L2_SIZE] = vector[HW_GENE.L2_SIZE]
    hw_gene[HW_GENE.L1_SIZE] = vector[HW_GENE.L1_SIZE]
    hw_gene[HW_GENE.NUM_PE] = vector[HW_GENE.NUM_PE]
    hw_gene[HW_GENE.BW] = vector[HW_GENE.BW]
    hw_gene[HW_GENE.NUM_DIM] = vector[HW_GENE.NUM_DIM]
    hw_gene[HW_GENE.DIM_SIZE_0] = vector[HW_GENE.DIM_SIZE_0]
    hw_gene[HW_GENE.DIM_SIZE_1] = vector[HW_GENE.DIM_SIZE_1]
    hw_gene[HW_GENE.DIM_SIZE_2] = vector[HW_GENE.DIM_SIZE_2]
    hw_gene[HW_GENE.PAR_DIM_K] = vector[HW_GENE.PAR_DIM_K]
    hw_gene[HW_GENE.PAR_DIM_C] = vector[HW_GENE.PAR_DIM_C]
    hw_gene[HW_GENE.PAR_DIM_Y] = vector[HW_GENE.PAR_DIM_Y]
    hw_gene[HW_GENE.PAR_DIM_X] = vector[HW_GENE.PAR_DIM_X]
    hw_gene[HW_GENE.PAR_DIM_R] = vector[HW_GENE.PAR_DIM_R]
    hw_gene[HW_GENE.PAR_DIM_S] = vector[HW_GENE.PAR_DIM_S]
    hw_gene[HW_GENE.GROUP_DENSITY] = vector[HW_GENE.GROUP_DENSITY]
    hw_gene[HW_GENE.BANK] = vector[HW_GENE.BANK]
    hw_gene[HW_GENE.L2_WEIGHT_SIZE] = vector[HW_GENE.L2_WEIGHT_SIZE]
    hw_gene[HW_GENE.L2_INPUT_SIZE] = vector[HW_GENE.L2_INPUT_SIZE]
    hw_gene[HW_GENE.L2_OUTPUT_SIZE] = vector[HW_GENE.L2_OUTPUT_SIZE]
    hw_gene[HW_GENE.L1_WEIGHT_SIZE] = vector[HW_GENE.L1_WEIGHT_SIZE]
    hw_gene[HW_GENE.L1_INPUT_SIZE] = vector[HW_GENE.L1_INPUT_SIZE]
    hw_gene[HW_GENE.L1_OUTPUT_SIZE] = vector[HW_GENE.L1_OUTPUT_SIZE]
    return hw_gene

def MapGene_mapper(vector):
    map_gene = [0 for i in range(len(MAPPING_GENE))]
    map_gene[MAPPING_GENE.ARR_LOOP_ORDER_K] = vector[MAPPING_GENE.ARR_LOOP_ORDER_K]
    map_gene[MAPPING_GENE.ARR_LOOP_ORDER_C] = vector[MAPPING_GENE.ARR_LOOP_ORDER_C]
    map_gene[MAPPING_GENE.ARR_LOOP_ORDER_Y] = vector[MAPPING_GENE.ARR_LOOP_ORDER_Y]
    map_gene[MAPPING_GENE.ARR_LOOP_ORDER_X] = vector[MAPPING_GENE.ARR_LOOP_ORDER_X]
    map_gene[MAPPING_GENE.ARR_LOOP_ORDER_R] = vector[MAPPING_GENE.ARR_LOOP_ORDER_R]
    map_gene[MAPPING_GENE.ARR_LOOP_ORDER_S] = vector[MAPPING_GENE.ARR_LOOP_ORDER_S]

    map_gene[MAPPING_GENE.ARR_TILE_SIZE_K] = vector[MAPPING_GENE.ARR_TILE_SIZE_K]
    map_gene[MAPPING_GENE.ARR_TILE_SIZE_C] = vector[MAPPING_GENE.ARR_TILE_SIZE_C]
    map_gene[MAPPING_GENE.ARR_TILE_SIZE_Y] = vector[MAPPING_GENE.ARR_TILE_SIZE_Y]
    map_gene[MAPPING_GENE.ARR_TILE_SIZE_X] = vector[MAPPING_GENE.ARR_TILE_SIZE_X]
    map_gene[MAPPING_GENE.ARR_TILE_SIZE_R] = vector[MAPPING_GENE.ARR_TILE_SIZE_R]
    map_gene[MAPPING_GENE.ARR_TILE_SIZE_S] = vector[MAPPING_GENE.ARR_TILE_SIZE_S]

    map_gene[MAPPING_GENE.PE_LOOP_ORDER_K] = vector[MAPPING_GENE.PE_LOOP_ORDER_K]
    map_gene[MAPPING_GENE.PE_LOOP_ORDER_C] = vector[MAPPING_GENE.PE_LOOP_ORDER_C]
    map_gene[MAPPING_GENE.PE_LOOP_ORDER_Y] = vector[MAPPING_GENE.PE_LOOP_ORDER_Y]
    map_gene[MAPPING_GENE.PE_LOOP_ORDER_X] = vector[MAPPING_GENE.PE_LOOP_ORDER_X]
    map_gene[MAPPING_GENE.PE_LOOP_ORDER_R] = vector[MAPPING_GENE.PE_LOOP_ORDER_R]
    map_gene[MAPPING_GENE.PE_LOOP_ORDER_S] = vector[MAPPING_GENE.PE_LOOP_ORDER_S]

    map_gene[MAPPING_GENE.L1_TILE_SIZE_K] = vector[MAPPING_GENE.L1_TILE_SIZE_K]
    map_gene[MAPPING_GENE.L1_TILE_SIZE_C] = vector[MAPPING_GENE.L1_TILE_SIZE_C]
    map_gene[MAPPING_GENE.L1_TILE_SIZE_Y] = vector[MAPPING_GENE.L1_TILE_SIZE_Y]
    map_gene[MAPPING_GENE.L1_TILE_SIZE_X] = vector[MAPPING_GENE.L1_TILE_SIZE_X]
    map_gene[MAPPING_GENE.L1_TILE_SIZE_R] = vector[MAPPING_GENE.L1_TILE_SIZE_R]
    map_gene[MAPPING_GENE.L1_TILE_SIZE_S] = vector[MAPPING_GENE.L1_TILE_SIZE_S]
    return map_gene


def HWGene_inverse_mapper(vector):
    cma_hw_gene = [0 for i in range(len(HW_GENE))]
    cma_hw_gene[HW_GENE.L2_SIZE] = vector[HW_GENE.L2_SIZE]
    cma_hw_gene[HW_GENE.L1_SIZE] = vector[HW_GENE.L1_SIZE]
    cma_hw_gene[HW_GENE.NUM_PE] = vector[HW_GENE.NUM_PE]
    cma_hw_gene[HW_GENE.BW] = vector[HW_GENE.BW]
    cma_hw_gene[HW_GENE.NUM_DIM] = vector[HW_GENE.NUM_DIM]
    cma_hw_gene[HW_GENE.DIM_SIZE_0] = vector[HW_GENE.DIM_SIZE_0]
    cma_hw_gene[HW_GENE.DIM_SIZE_1] = vector[HW_GENE.DIM_SIZE_1]
    cma_hw_gene[HW_GENE.DIM_SIZE_2] = vector[HW_GENE.DIM_SIZE_2]
    cma_hw_gene[HW_GENE.PAR_DIM_K] = vector[HW_GENE.PAR_DIM_K]
    cma_hw_gene[HW_GENE.PAR_DIM_C] = vector[HW_GENE.PAR_DIM_C]
    cma_hw_gene[HW_GENE.PAR_DIM_Y] = vector[HW_GENE.PAR_DIM_Y]
    cma_hw_gene[HW_GENE.PAR_DIM_X] = vector[HW_GENE.PAR_DIM_X]
    cma_hw_gene[HW_GENE.PAR_DIM_R] = vector[HW_GENE.PAR_DIM_R]
    cma_hw_gene[HW_GENE.PAR_DIM_S] = vector[HW_GENE.PAR_DIM_S]
    cma_hw_gene[HW_GENE.GROUP_DENSITY] = vector[HW_GENE.GROUP_DENSITY]
    cma_hw_gene[HW_GENE.BANK] = vector[HW_GENE.BANK]
    cma_hw_gene[HW_GENE.L2_WEIGHT_SIZE] = vector[HW_GENE.L2_WEIGHT_SIZE]
    cma_hw_gene[HW_GENE.L2_INPUT_SIZE] = vector[HW_GENE.L2_INPUT_SIZE]
    cma_hw_gene[HW_GENE.L2_OUTPUT_SIZE] = vector[HW_GENE.L2_OUTPUT_SIZE]
    cma_hw_gene[HW_GENE.L1_WEIGHT_SIZE] = vector[HW_GENE.L1_WEIGHT_SIZE]
    cma_hw_gene[HW_GENE.L1_INPUT_SIZE] = vector[HW_GENE.L1_INPUT_SIZE]
    cma_hw_gene[HW_GENE.L1_OUTPUT_SIZE] = vector[HW_GENE.L1_OUTPUT_SIZE]
    return cma_hw_gene

def MapGene_inverse_mapper(vector):
    cma_map_gene = [0 for i in range(len(MAPPING_GENE))]
    cma_map_gene[MAPPING_GENE.ARR_LOOP_ORDER_K] = vector[MAPPING_GENE.ARR_LOOP_ORDER_K]
    cma_map_gene[MAPPING_GENE.ARR_LOOP_ORDER_C] = vector[MAPPING_GENE.ARR_LOOP_ORDER_C]
    cma_map_gene[MAPPING_GENE.ARR_LOOP_ORDER_Y] = vector[MAPPING_GENE.ARR_LOOP_ORDER_Y]
    cma_map_gene[MAPPING_GENE.ARR_LOOP_ORDER_X] = vector[MAPPING_GENE.ARR_LOOP_ORDER_X]
    cma_map_gene[MAPPING_GENE.ARR_LOOP_ORDER_R] = vector[MAPPING_GENE.ARR_LOOP_ORDER_R]
    cma_map_gene[MAPPING_GENE.ARR_LOOP_ORDER_S] = vector[MAPPING_GENE.ARR_LOOP_ORDER_S]

    cma_map_gene[MAPPING_GENE.ARR_TILE_SIZE_K] = vector[MAPPING_GENE.ARR_TILE_SIZE_K]
    cma_map_gene[MAPPING_GENE.ARR_TILE_SIZE_C] = vector[MAPPING_GENE.ARR_TILE_SIZE_C]
    cma_map_gene[MAPPING_GENE.ARR_TILE_SIZE_Y] = vector[MAPPING_GENE.ARR_TILE_SIZE_Y]
    cma_map_gene[MAPPING_GENE.ARR_TILE_SIZE_X] = vector[MAPPING_GENE.ARR_TILE_SIZE_X]
    cma_map_gene[MAPPING_GENE.ARR_TILE_SIZE_R] = vector[MAPPING_GENE.ARR_TILE_SIZE_R]
    cma_map_gene[MAPPING_GENE.ARR_TILE_SIZE_S] = vector[MAPPING_GENE.ARR_TILE_SIZE_S]

    cma_map_gene[MAPPING_GENE.PE_LOOP_ORDER_K] = vector[MAPPING_GENE.PE_LOOP_ORDER_K]
    cma_map_gene[MAPPING_GENE.PE_LOOP_ORDER_C] = vector[MAPPING_GENE.PE_LOOP_ORDER_C]
    cma_map_gene[MAPPING_GENE.PE_LOOP_ORDER_Y] = vector[MAPPING_GENE.PE_LOOP_ORDER_Y]
    cma_map_gene[MAPPING_GENE.PE_LOOP_ORDER_X] = vector[MAPPING_GENE.PE_LOOP_ORDER_X]
    cma_map_gene[MAPPING_GENE.PE_LOOP_ORDER_R] = vector[MAPPING_GENE.PE_LOOP_ORDER_R]
    cma_map_gene[MAPPING_GENE.PE_LOOP_ORDER_S] = vector[MAPPING_GENE.PE_LOOP_ORDER_S]

    cma_map_gene[MAPPING_GENE.L1_TILE_SIZE_K] = vector[MAPPING_GENE.L1_TILE_SIZE_K]
    cma_map_gene[MAPPING_GENE.L1_TILE_SIZE_C] = vector[MAPPING_GENE.L1_TILE_SIZE_C]
    cma_map_gene[MAPPING_GENE.L1_TILE_SIZE_Y] = vector[MAPPING_GENE.L1_TILE_SIZE_Y]
    cma_map_gene[MAPPING_GENE.L1_TILE_SIZE_X] = vector[MAPPING_GENE.L1_TILE_SIZE_X]
    cma_map_gene[MAPPING_GENE.L1_TILE_SIZE_R] = vector[MAPPING_GENE.L1_TILE_SIZE_R]
    cma_map_gene[MAPPING_GENE.L1_TILE_SIZE_S] = vector[MAPPING_GENE.L1_TILE_SIZE_S]
    return cma_map_gene


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

if __name__ == "__main__":
    import pygad
    ga_instance = pygad.GA(
                   num_generations=num_gen,
                   num_parents_mating=num_parents,
                   fitness_func=get_a_fitness,
                   #initial_population=source_solutions,
                   sol_per_pop=num_pop,
                   num_genes=(len(HW_GENE) + len(MAPPING_GENE)*num_layers),
                   parent_selection_type="sss", #"sss",
                   #keep_parents=num_parents,
                   keep_elitism=num_parents,
                   crossover_type="single_point",
                   #mutation_type="random",
                   #mutation_percent_genes=10,
                   mutation_type="adaptive",
                   mutation_probability=[0.25, 0.1],
                   random_mutation_min_val=0,
                   random_mutation_max_val=1,
                   on_fitness=print_fitness)
    
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))

    '''
    optimizer = CMA(mean=(np.ones(len(HW_GENE)+len(MAPPING_GENE)) / 2), sigma=0.2)

    for generation in range(5):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            print(HWGene_mapper(x[0:len(HW_GENE)]))
            print(MapGene_mapper(x[len(HW_GENE):len(HW_GENE)+len(MAPPING_GENE)]))
            #value = quadratic(x[0], x[1])
            value = 0
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)
    '''
