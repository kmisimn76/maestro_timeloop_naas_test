

import numpy as np
import math

class Constraint_AlveoU200_Sparse:
    #def __init__(self, group_density=None, bank=None):
    def __init__(self, BRAM_max_size=600, DSP_max_size=256, group_density=4, bank=4):
        self.BRAM_max_size = BRAM_max_size # constraint for frequency performance
        self.DSP_max_size = DSP_max_size # constraint for freq. performance

        self.e_B = 1/18000 # 1 / (18Kb ?? / BRAM)
        self.align_byte = 4.5 # maximum BRAM bitwidth(Byte) to array reshape; 4.5 = 36bit(bram)/8(bit)
        self.L1_group_density = group_density
        self.L1_bank = bank
        self.L2_bank = bank

        # do not modify value (for auto-compute)
        self.L1wt_bank = 1
        self.L1wt_byte = 1
        self.L1wt_levg = 1 # multiplier
        self.L1in_bank = 1
        self.L1in_byte = 1
        self.L1in_levg = 1 # multiplier
        self.L1out_bank = 1
        self.L1out_byte = 4
        self.L1out_levg = 1 # multiplier
        self.L2wt_bank = 1
        self.L2wt_byte = 1
        self.L2wt_levg = 1 # multiplier
        self.L2in_bank = 1
        self.L2in_byte = 1
        self.L2in_levg = 1 # multiplier
        self.L2out_bank = 1
        self.L2out_byte = 4
        self.L2out_levg = 1 # multiplier
            
    def __compute_BRAM_usage(self, L1wt, L1in, L1out, L2wt, L2in, L2out):
        used_bram = 0 # used bram num
        for required, bank, byte, levg in zip([L1wt, L1in, L1out, L2wt, L2in, L2out],
                                              [self.L1wt_bank, self.L1in_bank, self.L1out_bank, self.L2wt_bank, self.L2in_bank, self.L2out_bank],
                                              [self.L1wt_byte, self.L1in_byte, self.L1out_byte, self.L2wt_byte, self.L2in_byte, self.L2out_byte],
                                              [self.L1wt_levg, self.L1in_levg, self.L1out_levg, self.L2wt_levg, self.L2in_levg, self.L2out_levg]):
            required, bank, byte, levg = float(required), float(bank), float(byte), float(levg)
            if bank==0 or levg==0: continue
            # w/ array reshape
            byte_alinged = math.ceil(byte/self.align_byte) * self.align_byte
            used_bram += math.ceil((required*byte_alinged)/bank*self.e_B) * (bank*(byte/self.align_byte)) * levg
            #print(required, bank, byte, levg)

            # w/o array reshape
            #used_bram += math.ceil((required*byte)/bank*self.e_B) * (bank) * levg
        return used_bram

    def __compute_DSP_usage(self, Xdim, Ydim):
        return Xdim * Ydim

    def set_constraint(self, BRAM_max_size=600, DSP_max_size=256, group_density=4, bank=4):
        self.BRAM_max_size = BRAM_max_size
        self.DSP_max_size = DSP_max_size
        self.L1_group_density = group_density
        self.L1_bank = bank
        self.L2_bank = bank

    def check_constraints(self, estimated, hw_info, map_info):
        #hw_info, map_info = gene2mapping(dimension, hw_gene, map_gene)
        hw_map = [1 for i in range(0,6)]  #dim order
        hw_map[hw_info.get_YDim()] = hw_info.get_Y()
        hw_map[hw_info.get_XDim()] = hw_info.get_X()
        K  = hw_map[0]
        C  = hw_map[1]
        WH = hw_map[2] if hw_map[3]==1 else hw_map[3] # W or H
        #print(hw_map,K,C,WH)
        stationary_type = 0
        # parameter example) WHKC, weight stationary
        in_bank = C if K==1 else (C if WH==1 else WH) #((K==1)?(0):((WH==1)?(C):(WH)))
        wt_bank = C if WH==1 else (K if C==1 else C)#((WH==1)?(0):((C==1)?(K):(C)))
        out_bank = K if C==1 else (WH if K==1 else K)#((C==1)?(0):((K==1)?(WH):(K)))
        levg_g = math.ceil((C if C!=1 else WH) / self.L1_group_density) # group num
        self.L1wt_bank = wt_bank * self.L1_bank
        self.L1in_bank = in_bank * self.L1_bank
        self.L1out_bank = out_bank * self.L1_bank
        self.L2wt_bank = wt_bank * self.L2_bank
        self.L2in_bank = in_bank * self.L2_bank
        self.L2out_bank = out_bank * self.L2_bank
        self.L1wt_levg = levg_g if C==1 else 1
        self.L1in_levg = 1
        self.L1out_levg = levg_g if C!=1 else 1
        ###
        used_bram = self.__compute_BRAM_usage(estimated['l1_weight'], estimated['l1_input'], estimated['l1_output'],
                                        estimated['l2_weight'], estimated['l2_input'], estimated['l2_output'])
        used_dsp = self.__compute_DSP_usage(hw_info.get_X(), hw_info.get_Y())
        #print(used_bram, used_dsp)
        if used_bram > self.BRAM_max_size: #Invalid
            return False
        if used_dsp > self.DSP_max_size: #Invalid
            return False
        return True #Valid

if __name__ == "__main__":
    from TimeloopMapping import *
    from HWGene import *
    from MapGene import *
    util = 1
    runtime = 1
    energy = 1
    area = 1
    mac = 1
    power = 1
    buffer_weight = 1
    buffer_input = 1
    buffer_output = 1
    l1_weight = 256
    l1_input = 4096
    l1_output = 4096
    l2_weight = 147456
    l2_input = 41472
    l2_output = 32768
    PEX=8
    PEY=8
    density = 2
    bank = 1
    constraint = Constraint_AlveoU200_Sparse(group_density=density, bank=bank)
    print(PEX,PEY,density, bank)
    estimated = {'util': util, 'cycle': runtime, 'energy': energy, 'area': area, 'mac': mac, 'power': power,
                  'buffer_weight': buffer_weight, 'buffer_input': buffer_input, 'buffer_output': buffer_output,
                  'l1_weight': l1_weight, 'l1_input': l1_input, 'l1_output': l1_output,
                  'l2_weight': l2_weight, 'l2_input': l2_input, 'l2_output': l2_output }
    hw_info = TIMELOOP_HW()
    hw_info.set_HW(PEX, PEY, 0, 1, 0, 0)
    mapping_info = None
    print(constraint.check_constraints(estimated, hw_info, mapping_info))
