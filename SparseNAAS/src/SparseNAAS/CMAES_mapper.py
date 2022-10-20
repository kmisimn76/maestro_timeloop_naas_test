import numpy as np
from cmaes import CMA

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
    hw_gene[HW_GENE.NUM_PE] = trunc_and_strength_quant(vector[HW_GENE.NUM_PE], MIN_PE, MAX_PE)
    hw_gene[HW_GENE.BW] = trunc_and_strength_quant(vector[HW_GENE.BW], 1, 1000)
    hw_gene[HW_GENE.NUM_DIM] = trunc_and_strength_quant(vector[HW_GENE.NUM_DIM], 2, 2) #FIXME
    hw_gene[HW_GENE.DIM_SIZE_0] = vector[HW_GENE.DIM_SIZE_0]
    hw_gene[HW_GENE.DIM_SIZE_1] = vector[HW_GENE.DIM_SIZE_1]
    hw_gene[HW_GENE.DIM_SIZE_2] = vector[HW_GENE.DIM_SIZE_2]
    hw_gene[HW_GENE.PAR_DIM_K] = trunc_and_strength_quant(vector[HW_GENE.PAR_DIM_K], 1, 6)
    hw_gene[HW_GENE.PAR_DIM_C] = trunc_and_strength_quant(vector[HW_GENE.PAR_DIM_C], 1, 6)
    hw_gene[HW_GENE.PAR_DIM_Y] = trunc_and_strength_quant(vector[HW_GENE.PAR_DIM_Y], 1, 6)
    hw_gene[HW_GENE.PAR_DIM_X] = trunc_and_strength_quant(vector[HW_GENE.PAR_DIM_X], 1, 6)
    hw_gene[HW_GENE.PAR_DIM_R] = trunc_and_strength_quant(vector[HW_GENE.PAR_DIM_R], 1, 6)
    hw_gene[HW_GENE.PAR_DIM_S] = trunc_and_strength_quant(vector[HW_GENE.PAR_DIM_S], 1, 6)
    hw_gene[HW_GENE.GROUP_DENSITY] = trunc_and_strength_quant(vector[HW_GENE.GROUP_DENSITY], 0, MAX_GROUP)
    hw_gene[HW_GENE.BANK] = trunc_and_strength_quant(vector[HW_GENE.BANK], MIN_BANK, 4)
    hw_gene[HW_GENE.L2_WEIGHT_SIZE] = trunc_and_strength_quant(vector[HW_GENE.L2_WEIGHT_SIZE], 100,4*128000)
    hw_gene[HW_GENE.L2_INPUT_SIZE] = trunc_and_strength_quant(vector[HW_GENE.L2_INPUT_SIZE], 100, 4*128000)
    hw_gene[HW_GENE.L2_OUTPUT_SIZE] = trunc_and_strength_quant(vector[HW_GENE.L2_OUTPUT_SIZE], 100, 4*128000)
    hw_gene[HW_GENE.L1_WEIGHT_SIZE] = vector[HW_GENE.L1_WEIGHT_SIZE]
    hw_gene[HW_GENE.L1_INPUT_SIZE] = vector[HW_GENE.L1_INPUT_SIZE]
    hw_gene[HW_GENE.L1_OUTPUT_SIZE] = vector[HW_GENE.L1_OUTPUT_SIZE]
    return hw_gene

def MapGene_mapper(vector):
    map_gene = [0 for i in range(len(MAPPING_GENE))]
    map_gene[MAPPING_GENE.ARR_LOOP_ORDER_K] = trunc_and_strength_quant(vector[MAPPING_GENE.ARR_LOOP_ORDER_K], 1, 6)
    map_gene[MAPPING_GENE.ARR_LOOP_ORDER_C] = trunc_and_strength_quant(vector[MAPPING_GENE.ARR_LOOP_ORDER_C], 1, 6)
    map_gene[MAPPING_GENE.ARR_LOOP_ORDER_Y] = trunc_and_strength_quant(vector[MAPPING_GENE.ARR_LOOP_ORDER_Y], 1, 6)
    map_gene[MAPPING_GENE.ARR_LOOP_ORDER_X] = trunc_and_strength_quant(vector[MAPPING_GENE.ARR_LOOP_ORDER_X], 1, 6)
    map_gene[MAPPING_GENE.ARR_LOOP_ORDER_R] = trunc_and_strength_quant(vector[MAPPING_GENE.ARR_LOOP_ORDER_R], 1, 6)
    map_gene[MAPPING_GENE.ARR_LOOP_ORDER_S] = trunc_and_strength_quant(vector[MAPPING_GENE.ARR_LOOP_ORDER_S], 1, 6)

    map_gene[MAPPING_GENE.ARR_TILE_SIZE_K] = vector[MAPPING_GENE.ARR_TILE_SIZE_K]
    map_gene[MAPPING_GENE.ARR_TILE_SIZE_C] = vector[MAPPING_GENE.ARR_TILE_SIZE_C]
    map_gene[MAPPING_GENE.ARR_TILE_SIZE_Y] = vector[MAPPING_GENE.ARR_TILE_SIZE_Y]
    map_gene[MAPPING_GENE.ARR_TILE_SIZE_X] = vector[MAPPING_GENE.ARR_TILE_SIZE_X]
    map_gene[MAPPING_GENE.ARR_TILE_SIZE_R] = vector[MAPPING_GENE.ARR_TILE_SIZE_R]
    map_gene[MAPPING_GENE.ARR_TILE_SIZE_S] = vector[MAPPING_GENE.ARR_TILE_SIZE_S]

    map_gene[MAPPING_GENE.PE_LOOP_ORDER_K] = trunc_and_strength_quant(vector[MAPPING_GENE.PE_LOOP_ORDER_K], 1, 6)
    map_gene[MAPPING_GENE.PE_LOOP_ORDER_C] = trunc_and_strength_quant(vector[MAPPING_GENE.PE_LOOP_ORDER_C], 1, 6)
    map_gene[MAPPING_GENE.PE_LOOP_ORDER_Y] = trunc_and_strength_quant(vector[MAPPING_GENE.PE_LOOP_ORDER_Y], 1, 6)
    map_gene[MAPPING_GENE.PE_LOOP_ORDER_X] = trunc_and_strength_quant(vector[MAPPING_GENE.PE_LOOP_ORDER_X], 1, 6)
    map_gene[MAPPING_GENE.PE_LOOP_ORDER_R] = trunc_and_strength_quant(vector[MAPPING_GENE.PE_LOOP_ORDER_R], 1, 6)
    map_gene[MAPPING_GENE.PE_LOOP_ORDER_S] = trunc_and_strength_quant(vector[MAPPING_GENE.PE_LOOP_ORDER_S], 1, 6)

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
    cma_hw_gene[HW_GENE.NUM_PE] = inverse_trunc_and_strength_quant(vector[HW_GENE.NUM_PE], MIN_PE, MAX_PE)
    cma_hw_gene[HW_GENE.BW] = inverse_trunc_and_strength_quant(vector[HW_GENE.BW], 1, 1000)
    cma_hw_gene[HW_GENE.NUM_DIM] = inverse_trunc_and_strength_quant(vector[HW_GENE.NUM_DIM], 2, 2) #FIXME
    cma_hw_gene[HW_GENE.DIM_SIZE_0] = vector[HW_GENE.DIM_SIZE_0]
    cma_hw_gene[HW_GENE.DIM_SIZE_1] = vector[HW_GENE.DIM_SIZE_1]
    cma_hw_gene[HW_GENE.DIM_SIZE_2] = vector[HW_GENE.DIM_SIZE_2]
    cma_hw_gene[HW_GENE.PAR_DIM_K] = inverse_trunc_and_strength_quant(vector[HW_GENE.PAR_DIM_K], 1, 6)
    cma_hw_gene[HW_GENE.PAR_DIM_C] = inverse_trunc_and_strength_quant(vector[HW_GENE.PAR_DIM_C], 1, 6)
    cma_hw_gene[HW_GENE.PAR_DIM_Y] = inverse_trunc_and_strength_quant(vector[HW_GENE.PAR_DIM_Y], 1, 6)
    cma_hw_gene[HW_GENE.PAR_DIM_X] = inverse_trunc_and_strength_quant(vector[HW_GENE.PAR_DIM_X], 1, 6)
    cma_hw_gene[HW_GENE.PAR_DIM_R] = inverse_trunc_and_strength_quant(vector[HW_GENE.PAR_DIM_R], 1, 6)
    cma_hw_gene[HW_GENE.PAR_DIM_S] = inverse_trunc_and_strength_quant(vector[HW_GENE.PAR_DIM_S], 1, 6)
    cma_hw_gene[HW_GENE.GROUP_DENSITY] = inverse_trunc_and_strength_quant(vector[HW_GENE.GROUP_DENSITY], 0, MAX_GROUP)
    cma_hw_gene[HW_GENE.BANK] = inverse_trunc_and_strength_quant(vector[HW_GENE.BANK], MIN_BANK, 4)
    cma_hw_gene[HW_GENE.L2_WEIGHT_SIZE] = inverse_trunc_and_strength_quant(vector[HW_GENE.L2_WEIGHT_SIZE], 100,4*128000)
    cma_hw_gene[HW_GENE.L2_INPUT_SIZE] = inverse_trunc_and_strength_quant(vector[HW_GENE.L2_INPUT_SIZE], 100, 4*128000)
    cma_hw_gene[HW_GENE.L2_OUTPUT_SIZE] = inverse_trunc_and_strength_quant(vector[HW_GENE.L2_OUTPUT_SIZE], 100, 4*128000)
    cma_hw_gene[HW_GENE.L1_WEIGHT_SIZE] = vector[HW_GENE.L1_WEIGHT_SIZE]
    cma_hw_gene[HW_GENE.L1_INPUT_SIZE] = vector[HW_GENE.L1_INPUT_SIZE]
    cma_hw_gene[HW_GENE.L1_OUTPUT_SIZE] = vector[HW_GENE.L1_OUTPUT_SIZE]
    return cma_hw_gene

def MapGene_inverse_mapper(vector):
    cma_map_gene = [0 for i in range(len(MAPPING_GENE))]
    cma_map_gene[MAPPING_GENE.ARR_LOOP_ORDER_K] = inverse_trunc_and_strength_quant(vector[MAPPING_GENE.ARR_LOOP_ORDER_K], 1, 6)
    cma_map_gene[MAPPING_GENE.ARR_LOOP_ORDER_C] = inverse_trunc_and_strength_quant(vector[MAPPING_GENE.ARR_LOOP_ORDER_C], 1, 6)
    cma_map_gene[MAPPING_GENE.ARR_LOOP_ORDER_Y] = inverse_trunc_and_strength_quant(vector[MAPPING_GENE.ARR_LOOP_ORDER_Y], 1, 6)
    cma_map_gene[MAPPING_GENE.ARR_LOOP_ORDER_X] = inverse_trunc_and_strength_quant(vector[MAPPING_GENE.ARR_LOOP_ORDER_X], 1, 6)
    cma_map_gene[MAPPING_GENE.ARR_LOOP_ORDER_R] = inverse_trunc_and_strength_quant(vector[MAPPING_GENE.ARR_LOOP_ORDER_R], 1, 6)
    cma_map_gene[MAPPING_GENE.ARR_LOOP_ORDER_S] = inverse_trunc_and_strength_quant(vector[MAPPING_GENE.ARR_LOOP_ORDER_S], 1, 6)

    cma_map_gene[MAPPING_GENE.ARR_TILE_SIZE_K] = vector[MAPPING_GENE.ARR_TILE_SIZE_K]
    cma_map_gene[MAPPING_GENE.ARR_TILE_SIZE_C] = vector[MAPPING_GENE.ARR_TILE_SIZE_C]
    cma_map_gene[MAPPING_GENE.ARR_TILE_SIZE_Y] = vector[MAPPING_GENE.ARR_TILE_SIZE_Y]
    cma_map_gene[MAPPING_GENE.ARR_TILE_SIZE_X] = vector[MAPPING_GENE.ARR_TILE_SIZE_X]
    cma_map_gene[MAPPING_GENE.ARR_TILE_SIZE_R] = vector[MAPPING_GENE.ARR_TILE_SIZE_R]
    cma_map_gene[MAPPING_GENE.ARR_TILE_SIZE_S] = vector[MAPPING_GENE.ARR_TILE_SIZE_S]

    cma_map_gene[MAPPING_GENE.PE_LOOP_ORDER_K] = inverse_trunc_and_strength_quant(vector[MAPPING_GENE.PE_LOOP_ORDER_K], 1, 6)
    cma_map_gene[MAPPING_GENE.PE_LOOP_ORDER_C] = inverse_trunc_and_strength_quant(vector[MAPPING_GENE.PE_LOOP_ORDER_C], 1, 6)
    cma_map_gene[MAPPING_GENE.PE_LOOP_ORDER_Y] = inverse_trunc_and_strength_quant(vector[MAPPING_GENE.PE_LOOP_ORDER_Y], 1, 6)
    cma_map_gene[MAPPING_GENE.PE_LOOP_ORDER_X] = inverse_trunc_and_strength_quant(vector[MAPPING_GENE.PE_LOOP_ORDER_X], 1, 6)
    cma_map_gene[MAPPING_GENE.PE_LOOP_ORDER_R] = inverse_trunc_and_strength_quant(vector[MAPPING_GENE.PE_LOOP_ORDER_R], 1, 6)
    cma_map_gene[MAPPING_GENE.PE_LOOP_ORDER_S] = inverse_trunc_and_strength_quant(vector[MAPPING_GENE.PE_LOOP_ORDER_S], 1, 6)

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
