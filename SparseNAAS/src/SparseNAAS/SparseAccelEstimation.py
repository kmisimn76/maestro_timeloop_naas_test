
from subprocess import Popen, PIPE
import yaml
import numpy as np
import os
import math

from HWGene import *
from MapGene import *

from SparseAccelEstimator.main import HW_DEF, MAPPING_DEF, calculateSparseAccel, checkLegalConstraintSparseAccel

df_dict = {1:"dla", 2:"shi", 3:"eye"}
m_type_dicts = {1:"CONV", 2:"DSCONV"}

############## ----------------------- ###################
############## ----------------------- ###################
# SparseAccelEstimatorr Perform Estimation
############## ----------------------- ###################
############## ----------------------- ###################

class SparseAccelEstimator():

    def __init__(self, random_file_name):
        self.random_file_name = random_file_name

    def softmax(self, a) :
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def gene2mapping(self, dimension, hw_gene, map_gene):
        hw_info = TIMELOOP_HW()
        mapping = TIMELOOP_MAPPING()

        hw_info.set_HW(hw_gene)
        mapping.set_mapping_gene(dimension[0:6], hw_info.dim_size, hw_gene, map_gene)
        # check invalid mapping
        if hw_info.get_XDim()>=4 or hw_info.get_YDim()>=4: #except R,S
            raise Exception('imported RS')
        if (hw_info.get_XDim()==2 and hw_info.get_YDim()==3) or \
           (hw_info.get_XDim()==3 and hw_info.get_YDim()==2): #except WH PE
            raise Exception('parallel WH')
        if mapping.get_mapping_PE_tile_size().count(1)==6:
            raise Exception('cannot exploit sparsity')

        return hw_info, mapping

    def adaptor_timeloop_to_sparseaccel(self,dimension, hw_info,mapping):
        #define HW paramemter
        X = hw_info.get_X()
        XDim = hw_info.get_XDim()
        Y = hw_info.get_Y()
        YDim = hw_info.get_YDim()
        ARRAY_K = X if XDim==DIM.K else (Y if YDim==DIM.K else 1)
        ARRAY_C = X if XDim==DIM.C else (Y if YDim==DIM.C else 1)
        ARRAY_W = X if XDim==DIM.X else (Y if YDim==DIM.X else 1)
        hw_def = HW_DEF(ARRAY_K, ARRAY_C, ARRAY_W, hw_info.get_bank(), hw_info.get_bank())

        #define Mapping parameter
        l2_loop_order_map = [-1, -1, -1, -1, -1, -1]
        l2_loop_order_map[mapping.get_mapping_l2_order()[0]] = 5
        l2_loop_order_map[mapping.get_mapping_l2_order()[1]] = 4
        l2_loop_order_map[mapping.get_mapping_l2_order()[2]] = 3
        l2_loop_order_map[mapping.get_mapping_l2_order()[3]] = 2
        l2_loop_order_map[mapping.get_mapping_l2_order()[4]] = 1
        l2_loop_order_map[mapping.get_mapping_l2_order()[5]] = 0
        l1_loop_order_map = [-1, -1, -1, -1, -1, -1]
        l1_loop_order_map[mapping.get_mapping_l1_order()[0]] = 5
        l1_loop_order_map[mapping.get_mapping_l1_order()[1]] = 4
        l1_loop_order_map[mapping.get_mapping_l1_order()[2]] = 3
        l1_loop_order_map[mapping.get_mapping_l1_order()[3]] = 2
        l1_loop_order_map[mapping.get_mapping_l1_order()[4]] = 1
        l1_loop_order_map[mapping.get_mapping_l1_order()[5]] = 0

        l1_map_size = [mapping.get_mapping_L1_tile_size()[i] for i in range(0,6)]  #dim order
        PE_map_size = [mapping.get_mapping_PE_tile_size()[i] for i in range(0,6)]  #dim order
        l1_map_size[hw_info.get_YDim()] *= PE_map_size[hw_info.get_YDim()] #mv invalid PE map size to l1 @ parallel dim
        l1_map_size[hw_info.get_XDim()] *= PE_map_size[hw_info.get_XDim()] #mv invalid PE map size to l1 @ parallel dim
        PE_map_size[hw_info.get_YDim()] = hw_info.get_Y()
        PE_map_size[hw_info.get_XDim()] = hw_info.get_X()
        sparsity_group_index = int(math.log(hw_info.get_group_density(), 2)) # group_density 1,2,4,8,16 -> idx 0,1,2,3,4
        density = float(dimension[6+sparsity_group_index]) if sparsity_group_index<=4 else 1.0 # density

        mapping_def = MAPPING_DEF(
         mapping.get_mapping_L2_tile_size()[0],mapping.get_mapping_L2_tile_size()[1],mapping.get_mapping_L2_tile_size()[2],mapping.get_mapping_L2_tile_size()[3],mapping.get_mapping_L2_tile_size()[4],mapping.get_mapping_L2_tile_size()[5],
         l2_loop_order_map[0],l2_loop_order_map[1],l2_loop_order_map[2],l2_loop_order_map[3],l2_loop_order_map[4],l2_loop_order_map[5],
         l1_map_size[0],l1_map_size[1],l1_map_size[2],l1_map_size[3],l1_map_size[4],l1_map_size[5],
         l1_loop_order_map[0],l1_loop_order_map[1],l1_loop_order_map[2],l1_loop_order_map[3],l1_loop_order_map[4],l1_loop_order_map[5],
         PE_map_size[0],PE_map_size[1],PE_map_size[2],PE_map_size[3],PE_map_size[4],PE_map_size[5], density)

        buffer_estimated = {'l1_weight': mapping_def.K_L1 * mapping_def.C_L1 * mapping_def.R_L1 * mapping_def.S_L1,
                            'l1_input':  mapping_def.C_L1 * mapping_def.H_L1 * mapping_def.W_L1,
                            'l1_output': mapping_def.K_L1 * mapping_def.H_L1 * mapping_def.W_L1,
                            'l2_weight': mapping_def.K_L2 * mapping_def.C_L2 * mapping_def.R_L2 * mapping_def.S_L2,
                            'l2_input':  mapping_def.C_L2 * mapping_def.H_L2 * mapping_def.W_L2,
                            'l2_output': mapping_def.K_L2 * mapping_def.H_L2 * mapping_def.W_L2
                            }

        return hw_def, mapping_def, buffer_estimated

    def observe_timeloop(self, gen, dimension, hw_gene, map_gene, judge, target_constraint, thread_id=None, firsttime=False, multi=False):

        if multi==True:
            if (len(dimension) != len(hw_gene)) or (len(dimension) != len(map_gene)):
                raise "Invalid Argument"
            lim = len(dimension)
            raise 'len must be 1'
        else:
            lim = 1
            dimension = [dimension]
            hw_gene = [hw_gene]
            map_gene = [map_gene]
        if len(dimension[0]) < 8: #6 diemsion + density
            raise "Model def don't contain density information"
        result_dir = "./timeloop_result/result_{}/".format('' if thread_id is None else thread_id)

        if lim > 1: raise "lim value is exceeded"

        try:
            hw_info, mapping_info = self.gene2mapping(dimension[0], hw_gene[0], map_gene[0])
        except Exception as e:
            # Invalid mapping
            #print(e)
            return None, (None, None), None

        # Get timeloop result
        try:
            #run sparseaccel estimator
            hw_def, mapping_def, buffer_estimated = self.adaptor_timeloop_to_sparseaccel(dimension[0], hw_info, mapping_info)

            target_constraint.set_constraint(group_density=hw_info.get_group_density(), bank=hw_info.get_bank())
            fpga_constraint_check = target_constraint.check_constraints(buffer_estimated, hw_info, mapping_info) if (target_constraint is not None) else True
            sparse_accel_check_mapping = checkLegalConstraintSparseAccel(hw_def, mapping_def)
            if (sparse_accel_check_mapping is False) or (fpga_constraint_check is False): #Check constraint Invalid
                '''
                if gen>=10: # ignore resource check before fore generation
                    #print("Gene has out of FPGA resource, then raise error")
                    raise Exception('invalid map: constraint')
                else:
                    #print("Initial adv. although Gene has out of FPGA resource")
                    pass
                '''
                raise Exception('invalid map: constraint')

            result_cycle = calculateSparseAccel(hw_def, mapping_def)

            #Get metrics(observation)
            buffer_weight = 0 #timeloop_stat_item_px_buffer[3][2][0][0].text
            buffer_input = 0 #timeloop_stat_item_px_buffer[3][2][0][1].text
            buffer_output = 0 #timeloop_stat_item_px_buffer[3][2][0][2].text
            l1_weight = 0 #timeloop_stat_item_px_l1[3][2][0][0].text
            l1_input = 0 #timeloop_stat_item_px_l1[3][2][0][1].text
            l1_output = 0 #timeloop_stat_item_px_l1[3][2][0][2].text
            l2_weight = 0 #timeloop_stat_item_px_l2[3][2][0][0].text
            l2_input = 0 #timeloop_stat_item_px_l2[3][2][0][1].text
            l2_output = 0 #timeloop_stat_item_px_l2[3][2][0][2].text
            util = 0 #float(f.readline().split(' ')[1])  # utilization
            runtime = result_cycle #float(f.readline().split(' ')[1])  # cycles
            energy = 999 #float(f.readline().split(' ')[1])  # energy
            area = 0 #float(f.readline().split(' ')[1])  # area
            l2_size = l2_weight + l2_input + l2_output
            l1_size = l1_weight + l1_input + l1_output
            mac = 1 #unsupport
            power = 1 #unsupport
            observation = [runtime, 1, energy, area, l1_size, l2_size, mac, power]
            estimated = {'util': util, 'cycle': runtime, 'energy': energy, 'area': area, 'mac': mac, 'power': power,
                        'buffer_weight': buffer_weight, 'buffer_input': buffer_input, 'buffer_output': buffer_output,
                        'l1_weight': l1_weight, 'l1_input': l1_input, 'l1_output': l1_output,
                        'l2_weight': l2_weight, 'l2_input': l2_input, 'l2_output': l2_output }

            if runtime <= 0 or energy <= 0: #Invalid
                raise Exception('invalid map: runtime')

            return observation, judge(observation), estimated
        except Exception as e:
            #raise "compile err?"
            # Invalid!
            #print(e)
            return None, (None, None), None

    def save_timeloop_def(self, dimension, hw_gene, map_gene, hw_file_name, map_file_name, prob_file_name, sparse_file_name):
        lim = 1
        dimension = [dimension]
        hw_gene = [hw_gene]
        map_gene = [map_gene]
        if len(dimension[0]) < 8: #6 diemsion + density
            raise "Model def don't contain density information"

        process = []
        main_m_file = self.random_file_name
        for i in range(lim):
            # Run timeloop
            m_file = main_m_file+"{}".format(i)
            try:
                hw_info, mapping_info = self.gene2mapping(dimension[i], hw_gene[i], map_gene[i])
            except:
                # Invalid mapping
                return None, None, None, None
            hw_def, mapping_def, buffer_estimated = self.adaptor_timeloop_to_sparseaccel(dimension[i], hw_info, mapping_info)
            with open(hw_file_name, "w") as file:
                file.write(str(hw_def.ARRAY_K)+str(","))
                file.write(str(hw_def.ARRAY_C)+str(","))
                file.write(str(hw_def.ARRAY_W)+str("\n"))
                file.write(str(mapping_def.L2_TILENUM_K)+str(","))
                file.write(str(mapping_def.L2_TILENUM_C)+str(","))
                file.write(str(mapping_def.L2_TILENUM_H)+str(","))
                file.write(str(mapping_def.L2_TILENUM_W)+str(","))
                file.write(str(mapping_def.L2_TILENUM_R)+str(","))
                file.write(str(mapping_def.L2_TILENUM_S)+str("\n"))
                file.write(str(mapping_def.L2_ORDER_K)+str(","))
                file.write(str(mapping_def.L2_ORDER_C)+str(","))
                file.write(str(mapping_def.L2_ORDER_H)+str(","))
                file.write(str(mapping_def.L2_ORDER_W)+str(","))
                file.write(str(mapping_def.L2_ORDER_R)+str(","))
                file.write(str(mapping_def.L2_ORDER_S)+str("\n"))
                file.write(str(mapping_def.L1_TILENUM_K)+str(","))
                file.write(str(mapping_def.L1_TILENUM_C)+str(","))
                file.write(str(mapping_def.L1_TILENUM_H)+str(","))
                file.write(str(mapping_def.L1_TILENUM_W)+str(","))
                file.write(str(mapping_def.L1_TILENUM_R)+str(","))
                file.write(str(mapping_def.L1_TILENUM_S)+str("\n"))
                file.write(str(mapping_def.L1_ORDER_K)+str(","))
                file.write(str(mapping_def.L1_ORDER_C)+str(","))
                file.write(str(mapping_def.L1_ORDER_H)+str(","))
                file.write(str(mapping_def.L1_ORDER_W)+str(","))
                file.write(str(mapping_def.L1_ORDER_R)+str(","))
                file.write(str(mapping_def.L1_ORDER_S)+str("\n"))
                file.write(str(mapping_def.TILE_K)+str(","))
                file.write(str(mapping_def.TILE_C)+str(","))
                file.write(str(mapping_def.TILE_H)+str(","))
                file.write(str(mapping_def.TILE_W)+str(","))
                file.write(str(mapping_def.TILE_R)+str(","))
                file.write(str(mapping_def.TILE_S)+str("\n"))

    def get_gene_HW_info(self, dimension, hw_gene, map_gene):
        hw_info, mapping_info = self.gene2mapping(dimension, hw_gene, map_gene)
        return [DIM.to_str(hw_info.get_XDim()), hw_info.get_X(), \
                DIM.to_str(hw_info.get_YDim()), hw_info.get_Y(), \
                hw_info.get_group_density(), hw_info.get_bank()]
