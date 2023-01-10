

import numpy as np
import math

class Constraint_AlveoU200_Sparse:
    #def __init__(self, BRAM_max_size=2800, threshold_use_LUTRAM=0, DSP_max_size=8196, group_density=4, bank=4):
    #def __init__(self, BRAM_max_size=1400, threshold_use_LUTRAM=0, DSP_max_size=8196, group_density=4, bank=4): #BRAM : (MAX DSP / (32x32=1024 PE)) * MAX MEM ~= 1400 BRAM
    #def __init__(self, BRAM_max_size=140, threshold_use_LUTRAM=2000, DSP_max_size=256, group_density=4, bank=4):
    #def __init__(self, BRAM_max_size=700//2, URAM_max_size=320//2, threshold_use_LUTRAM=0, DSP_max_size=8196, group_density=4, bank=4): #BRAM : (MAX DSP / (32x32=1024 PE)) * MAX MEM ~= 1400 BRAM
    #def __init__(self, BRAM_max_size=700, URAM_max_size=300, threshold_use_LUTRAM=0, DSP_max_size=8196, group_density=4, bank=4): #BRAM : (MAX DSP / (32x32=1024 PE)) * MAX MEM ~= 1400 BRAM
    #def __init__(self, BRAM_max_size=800, URAM_max_size=280, threshold_use_LUTRAM=0, DSP_max_size=8196, group_density=4, bank=4): #BRAM : (MAX DSP / (32x32=1024 PE)) * MAX MEM ~= 1400 BRAM
    #def __init__(self, BRAM_max_size=1440, URAM_max_size=640, threshold_use_LUTRAM=0, DSP_max_size=8196, group_density=4, bank=4): #BRAM : (MAX DSP / (32x32=1024 PE)) * MAX MEM ~= 1400 BRAM
    #def __init__(self, BRAM_max_size=2000, URAM_max_size=840, threshold_use_LUTRAM=0, DSP_max_size=8196, group_density=4, bank=4): #BRAM : (MAX DSP / (32x32=1024 PE)) * MAX MEM ~= 1400 BRAM
    def __init__(self, BRAM_max_size=200000000, URAM_max_size=840000000, threshold_use_LUTRAM=0, DSP_max_size=8196, group_density=4, bank=4): #BRAM : (MAX DSP / (32x32=1024 PE)) * MAX MEM ~= 1400 BRAM
        self.BRAM_max_size = BRAM_max_size # constraint for frequency performance
        self.URAM_max_size = URAM_max_size
        self.DSP_max_size = DSP_max_size # constraint for freq. performance

        #self.e_B = 1/(36000/8) # 1 / (18Kb ?? / BRAM)
        #self.align_byte = 4.5 # maximum BRAM bitwidth(Byte) to array reshape; 4.5 = 36bit(bram)/8(bit)
        self.bram_per_bit = 1/36000
        self.uram_per_bit = 1/288000
        self.bram_depth_choices = {"1bit": 32000, "2bit": 16000, "4bit": 8000, "8bit": 4000, "16bit": 2000, "32bit": 1000}
        self.uram_depth_choices = {"always": 4000}

        self.L1_group_density = group_density
        self.L1_bank = bank
        self.L2_bank = bank
        self.threshold_use_LUTRAM = threshold_use_LUTRAM #byte

        # do not modify value (for auto-compute)
        scale = 1#2 #1=INT8 / 2=INT4
        out_scale = 4
        #scale = 0.25 # FP32
        #out_scale = 1 # FP32

        self.L1wt_bank = 1
        self.L1wt_byte = 1 / scale
        self.L1wt_levg = 1 # group, ..
        self.L1in_bank = 1
        self.L1in_byte = 1 / scale
        self.L1in_levg = 1 # group, ..
        self.L1out_bank = 1
        self.L1out_byte = out_scale / scale
        self.L1out_levg = 1 # group, ..
        self.L2wt_bank = 1
        self.L2wt_byte = 1 / scale
        self.L2wt_levg = 1#2 # double buffering, ..
        self.L2in_bank = 1
        self.L2in_byte = 1 / scale
        self.L2in_levg = 1#2 # double buffering, ..
        self.L2out_bank = 1
        self.L2out_byte = out_scale / scale
        self.L2out_levg = 1#2 # double buffering, ..
            
    def __compute_BRAM_usage(self, L1wt, L1in, L1out, L2wt, L2in, L2out, log_=False):
        bram_usage = []
        uram_usage = []
        if log_:
            print("data_size\tparts\tdata/bank\t\tbits\tdatabit/part\tbram/part\t\tTot")
        for required, bank, byte, levg in zip([L1wt, L1in, L1out, L2wt, L2in, L2out],
                                              [self.L1wt_bank, self.L1in_bank, self.L1out_bank, self.L2wt_bank, self.L2in_bank, self.L2out_bank],
                                              [self.L1wt_byte, self.L1in_byte, self.L1out_byte, self.L2wt_byte, self.L2in_byte, self.L2out_byte],
                                              [self.L1wt_levg, self.L1in_levg, self.L1out_levg, self.L2wt_levg, self.L2in_levg, self.L2out_levg]):
            required, bank, byte, levg = float(required), float(bank), float(byte), float(levg)
            # july style
            '''
            if bank==0 or levg==0: continue
            data_size = required
            parts = bank
            data_per_part = data_size / parts
            bits = byte*8
            databit_per_part = data_per_part * bits
            bram_per_part = math.ceil(databit_per_part * self.bram_per_bit)
            total_bram = (parts * bram_per_part) * levg
            if log_:
                print("{}\t{}\t{}\t\t{}\t{}\t{}\t\t{}".format(data_size,parts,data_per_part,bits,databit_per_part, bram_per_part,total_bram))
            bram_usage.append(total_bram)
            '''
            # september style
            '''
             FIXME! L2*_levg must be modified for double buffering
            if bank==0 or levg==0: continue
            total_bram = math.ceil( (required*(byte*8)*self.bram_per_bit) / bank ) * bank * levg
            if required*levg < self.threshold_use_LUTRAM:
                total_bram = 0
            bram_usage.append(total_bram)
            '''
            # november style - considering uram, considering RAM depth
            bram_depth = self.bram_depth_choices['32bit' if byte==4 else ('16bit' if byte==2 else ('8bit' if byte==1 else ('4bit' if byte==0.5 else '1bit')))]
            uram_depth = self.uram_depth_choices['always']
            total_bram = 2 * bank * levg * math.ceil( math.ceil(required/bank) / bram_depth) if bank != 0 else 0
            total_uram = 2 * bank * levg * math.ceil( math.ceil(required/bank) / uram_depth) if bank != 0 else 0
            if required*levg < self.threshold_use_LUTRAM:
                total_bram = 0
                total_uram = 0
            bram_usage.append(total_bram)
            uram_usage.append(total_uram)
        return bram_usage, uram_usage
        #if log_:
        #    print(used_bram, used_uram)
        #return used_bram, used_uram, selected_ram_type

    def __compute_DSP_usage(self, Xdim, Ydim):
        return Xdim * Ydim

    def set_constraint(self, BRAM_max_size=-1, DSP_max_size=-1, group_density=-1, bank=-1):
        self.BRAM_max_size = BRAM_max_size if BRAM_max_size != -1 else self.BRAM_max_size
        self.DSP_max_size = DSP_max_size if DSP_max_size != -1 else self.DSP_max_size
        self.L1_group_density = group_density if group_density != -1 else self.L1_group_density
        self.L1_bank = bank if bank != -1 else self.L1_bank
        self.L2_bank = bank if bank != -1 else self.L2_bank

    def check_constraints(self, estimated, hw_info, map_info, use_sparsity, log_=False):
        #hw_info, map_info = gene2mapping(dimension, hw_gene, map_gene)
        hw_map = [1 for i in range(0,6)]  #dim order
        hw_map[hw_info.get_XDim()] = hw_info.get_X()
        if hw_info.get_YDim() is not None:
            hw_map[hw_info.get_YDim()] = hw_info.get_Y()
        if hw_info.get_ZDim() is not None:
            hw_map[hw_info.get_ZDim()] = hw_info.get_Z()
        K  = hw_map[0]
        C  = hw_map[1]
        #WH = hw_map[2] if hw_map[3]==1 else hw_map[3] # W or H
        WH = hw_map[2] * hw_map[3] # W * H
        WEIGHT_STATIONARY = 0
        OUTPUT_STATIONARY = 1
        INPUT_STATIONARY = 2
        stationary_type = 0 if WH==1 else (1 if C==1 else 2) # 0 - weight stationary
                                                             # 1 - output stationary
                                                             # 2 - input stationary
        if stationary_type == WEIGHT_STATIONARY:
            in_bank = C
            wt_bank = C
            out_bank= K
            levg_g = math.ceil(C / self.L1_group_density) # group num
            self.L1wt_bank = 0
            self.L1in_bank = in_bank * self.L1_bank 
            self.L1out_bank = out_bank * self.L1_bank 
            self.L2wt_bank = wt_bank #
            self.L2in_bank = in_bank * self.L2_bank
            self.L2out_bank = out_bank * self.L2_bank
            self.L1wt_levg = 1
            self.L1in_levg = 1
            self.L1out_levg = levg_g
        elif stationary_type == OUTPUT_STATIONARY:
            in_bank = WH
            wt_bank = K
            out_bank= WH
            levg_g = math.ceil(WH / self.L1_group_density) # group num
            self.L1wt_bank = wt_bank * self.L1_bank
            self.L1in_bank = in_bank * self.L1_bank 
            self.L1out_bank = 0
            self.L2wt_bank = wt_bank * self.L2_bank
            self.L2in_bank = in_bank * self.L2_bank
            self.L2out_bank = out_bank * self.L2_bank
            self.L1wt_levg = levg_g/2
            self.L1in_levg = 1
            self.L1out_levg = 1
        elif stationary_type == INPUT_STATIONARY:
            raise "err"
        '''
        # parameter example) WHKC, weight stationary
        in_bank = C if stationary_type==INPUT_STATIONARY  else (C if stationary_type==WEIGHT_STATIONARY else WH)
        wt_bank = C if stationary_type==WEIGHT_STATIONARY else (K if stationary_type==OUTPUT_STATIONARY else C)
        out_bank= WH if stationary_type==OUTPUT_STATIONARY else (WH if stationary_type==INPUT_STATIONARY else K)
        levg_g = math.ceil((C if stationary_type!=OUTPUT_STATIONARY else WH) / self.L1_group_density) # group num
        #levg_g = 1 #FIXME: for Dense
        self.L1wt_bank = wt_bank * self.L1_bank if stationary_type!=WEIGHT_STATIONARY else 0
        self.L1in_bank = in_bank * self.L1_bank if stationary_type!=INPUT_STATIONARY else 0
        self.L1out_bank = out_bank * self.L1_bank if stationary_type!=OUTPUT_STATIONARY else 0
        self.L2wt_bank = wt_bank * self.L2_bank
        self.L2in_bank = in_bank * self.L2_bank
        self.L2out_bank = out_bank * self.L2_bank
        #self.L1wt_levg = levg_g if stationary_type==OUTPUT_STATIONARY else 1
        self.L1wt_levg = max(1,levg_g/2) if stationary_type==OUTPUT_STATIONARY else 1 #BRAM dual-port optimized
        self.L1in_levg = 1
        self.L1out_levg = levg_g if stationary_type!=OUTPUT_STATIONARY else 1
        #if levg_g == 1 and self.L1_bank == 1: #if dense, L1 is not implemented
        '''
        if use_sparsity is False: #if dense, L1 is not implemented
            self.L1wt_levg = 0
            self.L1in_levg = 0
            self.L1out_levg = 0
        try:
            bram_usage, uram_usage = self.__compute_BRAM_usage(estimated['l1_weight'], estimated['l1_input'], estimated['l1_output'], #buffer size of bytes
                                            estimated['l2_weight'], estimated['l2_input'], estimated['l2_output'], log_)
            used_dsp = 0#self.__compute_DSP_usage(hw_info.get_X(), hw_info.get_Y())

            used_bram = float("inf")
            used_uram = float("inf")
            selected_ram_type = ["BRAM", "BRAM", "BRAM", "BRAM", "BRAM", "BRAM"]
            if log_:
                print("Possible type")
            '''
            for w1 in ["URAM", "BRAM"]:
                for i1 in ["URAM", "BRAM"]:
                    for o1 in ["URAM", "BRAM"]:
                        for w2 in ["URAM", "BRAM"]:
                            for i2 in ["URAM", "BRAM"]:
                                for o2 in ["URAM", "BRAM"]:
                                    select_type = [w1, i1, o1, w2, i2, o2]
                                    tot_bram = sum([bram_usage[l] if select_type[l]=="BRAM" else 0 for l in range(6)])
                                    tot_uram = sum([uram_usage[l] if select_type[l]=="URAM" else 0 for l in range(6)])
                                    if tot_bram <= self.BRAM_max_size and tot_uram <= self.URAM_max_size:
                                        selected_ram_type = select_type
                                        used_bram = tot_bram
                                        used_uram = tot_uram
                                        if log_:
                                            print(select_type)
            '''
            tot_bram = sum([bram_usage[l] for l in range(6)])
            tot_uram = 0
            selected = [0, 0, 0, 0, 0, 0]
            stack = [5,4,3,2,1,0]
            sets = []
            if tot_bram <= self.BRAM_max_size and tot_uram <= self.URAM_max_size:
                selected_ram_type = selected
                used_bram = tot_bram
                used_uram = tot_uram
            while len(stack)>0:
                p = stack.pop()
                if p > 0: stack = stack + list(range(p-1, -1, -1))
                selected[p] = 1 - selected[p]
                if selected[p] == 1:
                    tot_bram -= bram_usage[p] #-bram
                    tot_uram += uram_usage[p] #+uram
                else:
                    tot_bram += bram_usage[p] #+bram
                    tot_uram -= uram_usage[p] #-uram

                if tot_bram <= self.BRAM_max_size and tot_uram <= self.URAM_max_size:
                    selected_ram_type = selected
                    used_bram = tot_bram
                    used_uram = tot_uram
                if log_ and False:
                    print(tot_bram, tot_uram)
                    tot_bram_ = sum([bram_usage[l] if selected[l]==0 else 0 for l in range(6)])
                    tot_uram_ = sum([uram_usage[l] if selected[l]==1 else 0 for l in range(6)])
                    print(tot_bram_, tot_uram_)
                    print(selected)
                    sets.append([d for d in selected])
            if log_:
                print()
        except Exception as e:
            print(e)
            exit()
        if log_:
            print(stationary_type)
            print(bram_usage, uram_usage)
            print("Selected:", selected_ram_type)
            print(used_bram, used_uram, used_dsp)
            print(hw_map, K,C,WH)
            print(wt_bank, in_bank, out_bank)
            print(self.L1wt_levg, self.L1in_levg, self.L1out_levg)
            print(self.L1wt_bank, self.L1in_bank, self.L1out_bank)
            print("Levg_g: ", levg_g)
            print("L1")
            print(self.L1wt_bank*self.L1wt_levg, self.L1in_bank*self.L1in_levg, self.L1out_bank*self.L1out_levg)
            print("L2")
            print(self.L2wt_bank, self.L2in_bank, self.L2out_bank)
        if used_bram > self.BRAM_max_size or used_uram > self.URAM_max_size: #Invalid
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
    '''
    l1_weight = 1036818#16384
    l1_input = 124604#4096
    l1_output = 51259#0
    l2_weight = 1127621#540672
    l2_input = 476073#114800
    l2_output = 66705#114688
    '''
    #buffer_size = [325354.0,103174.0,100485.0,101476.57529715206,41929.23974894088,75463.47859619794]
    #buffer_size = [680467.0, 206150.0, 110666.0, 439193.97725600615, 199614.36794367808, 92773.70611971058]
    buffer_size = [510623.5,95920.5,110595.5,92357.00276812905,41937.40361880476,102384.87095319758]
    l2_weight = buffer_size[0]
    l2_input = buffer_size[1]
    l2_output = buffer_size[2]
    l1_weight = buffer_size[3]
    l1_input = buffer_size[4]
    l1_output = buffer_size[5]
    PEX=64 #K if weight stationary , K if output stastionary
    PEY=16 #C if weight stastionary, W if output stastionary
    weight_stationary = False
    output_stationary = True
    input_stationary = False
    st_order = [5,6,4,3,2,1] if weight_stationary else ([5,4,6,3,2,1] if output_stationary else [4,5,6,3,2,1])
    density = 1#16#int(math.log(4, 2))
    density = int(math.log(density, 2))
    bank = 1#8
    use_sparsity = True#False#True
    gene = [3000, 100, int(math.log(PEX*PEY,2)), 50, 2, 0.5, 0.5, 0.5, st_order[0], st_order[1], st_order[2], st_order[3], st_order[4], st_order[5], density, bank, l2_weight ,l2_input, l2_output, l1_weight, l1_input, l1_output, use_sparsity]

    constraint = Constraint_AlveoU200_Sparse(group_density=density, bank=bank)
    estimated = {'util': util, 'cycle': runtime, 'energy': energy, 'area': area, 'mac': mac, 'power': power,
                  'buffer_weight': buffer_weight, 'buffer_input': buffer_input, 'buffer_output': buffer_output,
                  'l1_weight': l1_weight, 'l1_input': l1_input, 'l1_output': l1_output,
                  'l2_weight': l2_weight, 'l2_input': l2_input, 'l2_output': l2_output }
    hw_info = TIMELOOP_HW()
    hw_info.set_HW(gene)
    hw_info.X=PEX
    hw_info.Y=PEY
    mapping_info = None
    constraint.set_constraint(group_density=2**density, bank=2**bank)
    print("HW info")
    print(PEX,PEY,2**density, bank)
    print(hw_info.get_X(), hw_info.get_Y())
    print(constraint.check_constraints(estimated, hw_info, mapping_info, use_sparsity, log_=True))
