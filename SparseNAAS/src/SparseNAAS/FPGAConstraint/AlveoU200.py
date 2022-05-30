

import numpy as np
import math

class Constraint_AlveoU200_Sparse:
    #def __init__(self, group_density=None, bank=None):
    def __init__(self, BRAM_max_size=600, DSP_max_size=256, group_density=4, bank=4):
        self.BRAM_max_size = BRAM_max_size # constraint for frequency performance
        self.DSP_max_size = DSP_max_size # constraint for freq. performance

        self.e_B = 1/(36000/8) # 1 / (18Kb ?? / BRAM)
        self.align_byte = 4.5 # maximum BRAM bitwidth(Byte) to array reshape; 4.5 = 36bit(bram)/8(bit)
        self.bram_per_bit = 1/36000
        self.L1_group_density = group_density
        self.L1_bank = bank
        self.L2_bank = bank

        # do not modify value (for auto-compute)
        self.L1wt_bank = 1
        self.L1wt_byte = 1
        self.L1wt_levg = 1 # group, ..
        self.L1in_bank = 1
        self.L1in_byte = 1
        self.L1in_levg = 1 # group, ..
        self.L1out_bank = 1
        self.L1out_byte = 4
        self.L1out_levg = 1 # group, ..
        self.L2wt_bank = 1
        self.L2wt_byte = 1
        self.L2wt_levg = 2 # double buffering, ..
        self.L2in_bank = 1
        self.L2in_byte = 1
        self.L2in_levg = 2 # double buffering, ..
        self.L2out_bank = 1
        self.L2out_byte = 4
        self.L2out_levg = 2 # double buffering, ..
            
    def __compute_BRAM_usage(self, L1wt, L1in, L1out, L2wt, L2in, L2out, log_=False):
        used_bram = 0 # used bram num
        #if log_:
        #    print("array size\t# bank\tdata byte\t\tcompress byte\tleverage\tbram per bank")
        if log_:
            print("data_size\tparts\tdata/bank\t\tbits\tdatabit/part\tbram/part\t\tTot")
        for required, bank, byte, levg in zip([L1wt, L1in, L1out, L2wt, L2in, L2out],
                                              [self.L1wt_bank, self.L1in_bank, self.L1out_bank, self.L2wt_bank, self.L2in_bank, self.L2out_bank],
                                              [self.L1wt_byte, self.L1in_byte, self.L1out_byte, self.L2wt_byte, self.L2in_byte, self.L2out_byte],
                                              [self.L1wt_levg, self.L1in_levg, self.L1out_levg, self.L2wt_levg, self.L2in_levg, self.L2out_levg]):
            required, bank, byte, levg = float(required), float(bank), float(byte), float(levg)
            if bank==0 or levg==0: continue
            # w/ array reshape
            '''
            # old style
            byte_alinged = math.ceil(byte/self.align_byte) * self.align_byte
            used_bram += math.ceil((required*byte_alinged)/bank*self.e_B) * (bank*(byte/self.align_byte)) * levg
            if log_:
                print("{}\t{}\t{}\t\t{}\t{}\t{}".format(required, bank, byte, byte_alinged, levg, math.ceil((required*byte_alinged)/bank*self.e_B)))
            '''
            # new style
            data_size = required
            parts = bank
            data_per_part = data_size / parts
            bits = byte*8
            databit_per_part = data_per_part * bits
            bram_per_part = math.ceil(databit_per_part * self.bram_per_bit)
            total_bram = (parts * bram_per_part) * levg
            if log_:
                print("{}\t{}\t{}\t\t{}\t{}\t{}\t\t{}".format(data_size,parts,data_per_part,bits,databit_per_part, bram_per_part,total_bram))
            used_bram += total_bram

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

    def check_constraints(self, estimated, hw_info, map_info, log_=False):
        #hw_info, map_info = gene2mapping(dimension, hw_gene, map_gene)
        hw_map = [1 for i in range(0,6)]  #dim order
        hw_map[hw_info.get_YDim()] = hw_info.get_Y()
        hw_map[hw_info.get_XDim()] = hw_info.get_X()
        K  = hw_map[0]
        C  = hw_map[1]
        WH = hw_map[2] if hw_map[3]==1 else hw_map[3] # W or H
        #print(hw_map,K,C,WH)
        WEIGHT_STATIONARY = 0
        OUTPUT_STATIONARY = 1
        INPUT_STATIONARY = 2
        stationary_type = 0 if WH==1 else (1 if C==1 else 2) # 0 - weight stationary
                                                             # 1 - output stationary
                                                             # 2 - input stationary
        # parameter example) WHKC, weight stationary
        in_bank = C if stationary_type==INPUT_STATIONARY  else (C if stationary_type==WEIGHT_STATIONARY else WH) #((K==1)?(1):((WH==1)?(C):(WH)))
        wt_bank = C if stationary_type==WEIGHT_STATIONARY else (K if stationary_type==OUTPUT_STATIONARY else C)#((WH==1)?(1):((C==1)?(K):(C)))
        out_bank= WH if stationary_type==OUTPUT_STATIONARY else (WH if stationary_type==INPUT_STATIONARY else K)#((C==1)?(1):((K==1)?(WH):(K)))
        levg_g = math.ceil((C if stationary_type!=OUTPUT_STATIONARY else WH) / self.L1_group_density) # group num
        self.L1wt_bank = wt_bank * self.L1_bank
        self.L1in_bank = in_bank * self.L1_bank
        self.L1out_bank = out_bank * self.L1_bank
        self.L2wt_bank = wt_bank * self.L2_bank
        self.L2in_bank = in_bank * self.L2_bank
        self.L2out_bank = out_bank * self.L2_bank
        self.L1wt_levg = levg_g if stationary_type==OUTPUT_STATIONARY else 1
        self.L1in_levg = 1
        self.L1out_levg = levg_g if stationary_type!=OUTPUT_STATIONARY else 1
        ###
        used_bram = self.__compute_BRAM_usage(estimated['l1_weight'], estimated['l1_input'], estimated['l1_output'],
                                        estimated['l2_weight'], estimated['l2_input'], estimated['l2_output'], log_)
        used_dsp = self.__compute_DSP_usage(hw_info.get_X(), hw_info.get_Y())
        if log_:
            print(stationary_type)
            print(used_bram, used_dsp)
            print(hw_map, K,C,WH)
            print(wt_bank, in_bank, out_bank)
            print(self.L1wt_levg, self.L1in_levg, self.L1out_levg)
            print(self.L1wt_bank, self.L1in_bank, self.L1out_bank)
            print("L1")
            print(self.L1wt_bank*self.L1wt_levg, self.L1in_bank*self.L1in_levg, self.L1out_bank*self.L1out_levg)
            print("L2")
            print(self.L2wt_bank, self.L2in_bank, self.L2out_bank)
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
    l1_weight = 16384#256
    l1_input = 4096#409
    l1_output = 0#409
    l2_weight = 540672#14745
    l2_input = 114800#4305#41472
    l2_output = 114688#4014#32768
    PEX=32
    PEY=8
    weight_stationary = False
    output_stationary = True
    input_stationary = False
    st_order = [6,5,4,3,2,1] if weight_stationary else ([5,4,6,3,2,1] if output_stationary else [4,5,6,3,2,1])
    density = int(math.log(4, 2)) #int(math.log(4, 2))
    bank = 8
    constraint = Constraint_AlveoU200_Sparse(group_density=density, bank=bank)
    estimated = {'util': util, 'cycle': runtime, 'energy': energy, 'area': area, 'mac': mac, 'power': power,
                  'buffer_weight': buffer_weight, 'buffer_input': buffer_input, 'buffer_output': buffer_output,
                  'l1_weight': l1_weight, 'l1_input': l1_input, 'l1_output': l1_output,
                  'l2_weight': l2_weight, 'l2_input': l2_input, 'l2_output': l2_output }
    hw_info = TIMELOOP_HW()
    gene = [3000, 100, int(math.log(PEX*PEY,2)), 50, 2, 0.5, 0.5, 0.5, st_order[0], st_order[1], st_order[2], st_order[3], st_order[4], st_order[5], density, bank]
    hw_info.set_HW(gene)
    hw_info.X=PEX
    hw_info.Y=PEY
    mapping_info = None
    constraint.set_constraint(group_density=2**density, bank=bank)
    print(PEX,PEY,2**density, bank)
    print(hw_info.get_X(), hw_info.get_Y())
    print(constraint.check_constraints(estimated, hw_info, mapping_info, log_=True))
