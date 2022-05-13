

import numpy as np
import math
from TimeloopMapping import *

class Constraint_AlveoU200_Sparse:
    def __init__(self):
        self.BRAM_max_size = 600 # constraint for frequency performance
        self.DSP_max_size = 2000 # constraint for freq. performance

        self.e_B = 1/18000 # 1 / (18Kb ?? / BRAM)
        self.align_byte = 4.5 # maximum BRAM bitwidth(Byte) to array reshape; 4.5 = 36bit(bram)/8(bit)
        self.L1_group_density = 4
        self.L1_bank = 8
        self.L2_bank = 8

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
            
    def compute_BRAM_usage(self, L1wt, L1in, L1out, L2wt, L2in, L2out):
        used_bram = 0 # used bram num
        for required, bank, byte, levg in zip([L1wt, L1in, L1out, L2wt, L2in, L2out],
                                              [self.L1wt_bank, self.L1in_bank, self.L1out_bank, self.L2wt_bank, self.L2in_bank, self.L2out_bank],
                                              [self.L1wt_byte, self.L1in_byte, self.L1out_byte, self.L2wt_byte, self.L2in_byte, self.L2out_byte],
                                              [self.L1wt_levg, self.L1in_levg, self.L1out_levg, self.L2wt_levg, self.L2in_levg, self.L2out_levg]):
            # w/ array reshape
            byte_alinged = math.ceil(byte/self.align_byte) * self.align_byte
            used_bram += math.ceil((required*byte_alinged)/bank*self.e_B) * (bank*(byte/self.align_byte)) * levg

            # w/o array reshape
            #used_bram += math.ceil((required*byte)/bank*self.e_B) * (bank) * levg
        return used_bram

    def compute_DSP_usage(self, Xdim, Ydim):
        return Xdim * Ydim

    def check_constraints(self, estimated, hw_info, map_info):

        # parameter example) WHKC, weight stationary
        self.L1wt_bank = hw_info.get_X()
        self.L1in_bank = hw_info.get_X() * self.L1_bank
        self.L1out_bank = hw_info.get_Y() * self.L1_bank
        self.L2wt_bank = hw_info.get_X()
        self.L2in_bank = hw_info.get_X() * self.L2_bank
        self.L2out_bank = hw_info.get_Y() * self.L2_bank
        self.L1out_levg = math.ceil(hw_info.get_X() / self.L1_group_density) # group num
        ###

        used_bram = self.compute_BRAM_usage(estimated['l1_weight'], estimated['l1_input'], estimated['l1_output'],
                                        estimated['l2_weight'], estimated['l2_input'], estimated['l2_output'])
        used_dsp = self.compute_DSP_usage(hw_info.get_X(), hw_info.get_Y())
        print(used_bram, used_dsp)
        if used_bram > self.BRAM_max_size: #Invalid
            return False
        if used_dsp > self.DSP_max_size: #Invalid
            return False
        return True #Valid

if __name__ == "__main__":
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
    constraint = Constraint_AlveoU200_Sparse()
    estimated = {'util': util, 'cycle': runtime, 'energy': energy, 'area': area, 'mac': mac, 'power': power,
                  'buffer_weight': buffer_weight, 'buffer_input': buffer_input, 'buffer_output': buffer_output,
                  'l1_weight': l1_weight, 'l1_input': l1_input, 'l1_output': l1_output,
                  'l2_weight': l2_weight, 'l2_input': l2_input, 'l2_output': l2_output }
    hw_info = TIMELOOP_HW()
    hw_info.set_HW(16, 16, 0, 0, 0, 0)
    mapping_info = None
    print(constraint.check_constraints(estimated, hw_info, mapping_info))
