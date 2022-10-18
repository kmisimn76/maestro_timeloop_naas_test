
from subprocess import Popen, PIPE
import yaml
import numpy as np
import os

from HWGene import *
from MapGene import *
#from HWGene import DIM
#from MapGene import DIM
#import Mapping

df_dict = {1:"dla", 2:"shi", 3:"eye"}
m_type_dicts = {1:"CONV", 2:"DSCONV"}

############## ----------------------- ###################
############## ----------------------- ###################
# Sparseloop Perform Estimation
############## ----------------------- ###################
############## ----------------------- ###################

class SparseloopEstimator():

    def __init__(self, random_file_name):
        self.random_file_name = random_file_name
        self.example_yaml_hw = "timeloop_example_yaml/sparse_example/sparse_npu.arch.yaml"
        self.example_yaml_prob = "timeloop_example_yaml/sparse_example/conv1d_sparse.prob.yaml"
        self.example_yaml_map = "timeloop_example_yaml/sparse_example/sparse_npu.map.yaml"
        self.example_yaml_sparseopt = "timeloop_example_yaml/sparse_example/sparse_opt.yaml"
        self.result_yaml_hw = "../../data/timeloop/hw_.yaml"
        self.result_yaml_prob = "../../data/timeloop/problem_.yaml"
        self.result_yaml_map = "../../data/timeloop/mapping_.yaml"
        self.result_yaml_sparseopt = "../../data/timeloop/sparse_.yaml"

    def softmax(self, a) :
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def gene2mapping(self, dimension, hw_gene, map_gene):
        hw_info = TIMELOOP_HW()
        mapping = TIMELOOP_MAPPING()
        #hw_info.set_HW(dim_size[0], dim_size[1], selected_hw_dim[0][0], selected_hw_dim[1][0], hw_gene[HW_GENE.L2_SIZE], hw_gene[HW_GENE.L1_SIZE], hw_gene[HW_GENE.)
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


    def write_timeloop_hw(self, dimension, hw_info, mapping_info, filename_=None, layer_id=0):
        file_name = self.result_yaml_hw if filename_ is None else filename_
        example_file_name = self.example_yaml_hw
        with open(example_file_name, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        x_size = hw_info.get_X()
        y_size = hw_info.get_Y()

        data_DRAM = data['architecture']['subtree'][0]['local'][0]
        data_L2 = data['architecture']['subtree'][0]['subtree'][0]['local'][0]
        data_L1 = data['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local'][0]
        data_SIMDCore = data['architecture']['subtree'][0]['subtree'][0]['subtree'][0]
        data_PE_Y = data_SIMDCore['subtree'][0]
        data_PE_X = data_PE_Y['subtree'][0]
        data_PE_Buffer = data_PE_X['local'][0]
        data_PE_MAC = data_PE_X['local'][1]

        data_PE_Y['name'] = 'PErow[0..{}]'.format(max(y_size-1,1)) # Exception case(incur Timeloop error); when x=y=1
        data_PE_X['name'] = 'PE[0..{}]'.format(max(x_size-1,1)) # Exception case(incur Timeloop error); when x=y=1

        data['architecture']['bank'] = hw_info.get_bank()
        data['architecture']['group_density'] = hw_info.get_group_density()

        hw_map = [1 for i in range(0,6)]  #dim order
        hw_map[hw_info.get_YDim()] = hw_info.get_Y()
        hw_map[hw_info.get_XDim()] = hw_info.get_X()
        bank = hw_info.get_bank()

        #old
        #input_bandwidth  = max(hw_map[DIM.C], hw_map[DIM.X], hw_map[DIM.Y]) * (1 if hw_map[DIM.K]==1 else bank)
        #weight_bandwidth = max(hw_map[DIM.K], hw_map[DIM.C]) * (1 if (hw_map[DIM.X]==1 and hw_map[DIM.Y]==1) else bank)
        #output_bandwidth = max(hw_map[DIM.K], hw_map[DIM.X], hw_map[DIM.Y]) * (1 if hw_map[DIM.C]==1 else bank)
        #new
        WEIGHT_STATIONARY = True if (hw_map[DIM.X]==1 and hw_map[DIM.Y]==1) else False
        OUTPUT_STATIONARY = True if (hw_map[DIM.C]==1) else False
        INPUT_STATIONARY = True if (hw_map[DIM.K]==1) else False
        input_bandwidth  = (hw_map[DIM.C] if WEIGHT_STATIONARY else max(hw_map[DIM.X],hw_map[DIM.Y])) * (1 if INPUT_STATIONARY else bank)
        weight_bandwidth = (hw_map[DIM.C] if INPUT_STATIONARY  else hw_map[DIM.K]) * (1 if WEIGHT_STATIONARY else bank)
        output_bandwidth = (hw_map[DIM.K] if WEIGHT_STATIONARY else max(hw_map[DIM.X],hw_map[DIM.Y])) * (1 if OUTPUT_STATIONARY else bank)

        data_L2['attributes']['read_bandwidth'] = input_bandwidth + weight_bandwidth
        data_L2['attributes']['write_bandwidth'] = output_bandwidth

        '''
        burst_size = 4096
        transaction_delay = 75
        burst_delay = 2
        overhead = 2

        length = (mapping_info.get_mapping_L1_tile_size()[3]*mapping_info.get_mapping_PE_tile_size()[3]) #L1_size_W , for weight stationary
        packet_size = 1 * hw_map[DIM.C] #1byte precision, 
        iteration = (mapping_info.get_mapping_L1_tile_size()[2]*mapping_info.get_mapping_PE_tile_size()[2]) * \
                        (mapping_info.get_mapping_L1_tile_size()[1]*mapping_info.get_mapping_PE_tile_size()[1])/hw_map[DIM.C] #L1_size_H * L1_size_C/hw_C
        dram_input_bw = (length*packet_size*iteration) / ((length+((length*packet_size/burst_size)*burst_delay)+transaction_delay)*iteration + overhead)

        length = (mapping_info.get_mapping_L1_tile_size()[0]*mapping_info.get_mapping_PE_tile_size()[3]) * \
                    (mapping_info.get_mapping_L1_tile_size()[1]*mapping_info.get_mapping_PE_tile_size()[1]) * \
                    (mapping_info.get_mapping_L1_tile_size()[4]*mapping_info.get_mapping_PE_tile_size()[4]) * \
                    (mapping_info.get_mapping_L1_tile_size()[5]*mapping_info.get_mapping_PE_tile_size()[5]) / hw_map[DIM.K]  #L1_size_R*S*C*K
        packet_size = 1 * hw_map[DIM.K] #1byte precision, 
        iteration = 1
        dram_weight_bw = (length*packet_size*iteration) / ((length+((length*packet_size/burst_size)*burst_delay)+transaction_delay)*iteration + overhead)

        length = (mapping_info.get_mapping_L1_tile_size()[3]*mapping_info.get_mapping_PE_tile_size()[3]) #L1_size_W , for weight stationary
        packet_size = 4 * hw_map[DIM.K] #4byte precision, DIM K
        iteration = (mapping_info.get_mapping_L1_tile_size()[2]*mapping_info.get_mapping_PE_tile_size()[2]) * \
                        (mapping_info.get_mapping_L1_tile_size()[0]*mapping_info.get_mapping_PE_tile_size()[0])/hw_map[DIM.K] #L1_size_H * L1_size_K/hw_K
        dram_output_bw = (length*packet_size*iteration) / ((length+((length*packet_size/burst_size)*burst_delay)+transaction_delay)*iteration + overhead)

        data_DRAM['attributes']['read_bandwidth'] = dram_input_bw + dram_weight_bw
        data_DRAM['attributes']['write_bandwidth'] = dram_output_bw
        '''
        raise "need to copy code from TimeloopEstimator.py"

        with open(file_name, "w") as f:
            yaml.dump(data, f)
        return file_name

    def write_timeloop_problem(self, dimension, hw_info, mapping_info, filename_=None, layer_id=0):
        file_name = self.result_yaml_prob if filename_ is None else filename_
        example_file_name = self.example_yaml_prob
        with open(example_file_name, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        l2_size_ = mapping_info.get_mapping_L2_tile_size()
        l1_size_ = mapping_info.get_mapping_L1_tile_size()
        pe_size_ = mapping_info.get_mapping_PE_tile_size()
        par_size_ = mapping_info.get_mapping_parallel_size()
        dim_size_ = [i*j*l*k for (i,j,l,k) in zip(l2_size_, l1_size_, pe_size_, par_size_)]
        #print(l2_size_, l1_size_, pe_size_, par_size_, dim_size_, mapping_info.mapping_gene_raw, mapping_info.mapping_tile_size, mapping_info.mapping_inner_tile_size)
        for i in range(0,6):
            if dim_size_[i] < dimension[i]:
                print("Dim size must be large than(or eqaul to) problem dimension: ", dim_size_[i], dimension[i])
                raise 'Mapping Err'
        data['problem']['instance']['M'] = int(dim_size_[0])
        data['problem']['instance']['C'] = int(dim_size_[1])
        data['problem']['instance']['P'] = int(dim_size_[2])
        data['problem']['instance']['Q'] = int(dim_size_[3])
        data['problem']['instance']['R'] = int(dim_size_[4])
        data['problem']['instance']['S'] = int(dim_size_[5])
        sparsity_group_index = int(math.log(hw_info.get_group_density(), 2)) # group_density 1,2,4,8,16 -> idx 0,1,2,3,4
        data['problem']['instance']['densities']['Inputs']['density'] = float(dimension[6+sparsity_group_index]) # density

        with open(file_name, "w") as f:
            yaml.dump(data, f)
        return file_name

    def write_timeloop_mapping(self, dimension, hw_info, mapping_info, filename_=None, layer_id=0):
        if len(dimension) > 11:
            m_type = m_type_dicts[int(dimension[-1])]
        else:
            m_type = "CONV"
        # Dataflow description
        file_name = self.result_yaml_map if filename_ is None else filename_
        example_file_name = self.example_yaml_map
        with open(example_file_name, "r") as f:
            mapping_data = yaml.load(f, Loader=yaml.FullLoader)
        mapping_data_DRAM_temp = mapping_data['mapping'][0]
        mapping_data_L2_temp = mapping_data['mapping'][1]
        mapping_data_L1_temp = mapping_data['mapping'][2]
        mapping_data_L1_spat = mapping_data['mapping'][3]
        mapping_data_Buf = mapping_data['mapping'][4]

        #DRAM map
        mapping_data_DRAM_temp['factors'] = "M={} C={} P={} Q={} R={} S={} N=1".format(mapping_info.get_mapping_L2_tile_size()[0],
                                                                                            mapping_info.get_mapping_L2_tile_size()[1],
                                                                                            mapping_info.get_mapping_L2_tile_size()[2],
                                                                                            mapping_info.get_mapping_L2_tile_size()[3],
                                                                                            mapping_info.get_mapping_L2_tile_size()[4],
                                                                                            mapping_info.get_mapping_L2_tile_size()[5])
        mapping_data_DRAM_temp['permutation'] = "N{}{}{}{}{}{}".format(DIM.to_str_timeloop(mapping_info.get_mapping_l2_order()[0]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_l2_order()[1]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_l2_order()[2]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_l2_order()[3]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_l2_order()[4]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_l2_order()[5]))
        #L2 map(=1)
        #mapping_data_L2_temp['factors'] = "M=1 C=1 P=1 Q=1 R=1 S=1 N=1"
        mapping_data_L2_temp['factors'] = "M={} C={} P={} Q={} R={} S={} N=1".format(mapping_info.get_mapping_L1_tile_size()[0],
                                                                                            mapping_info.get_mapping_L1_tile_size()[1],
                                                                                            mapping_info.get_mapping_L1_tile_size()[2],
                                                                                            mapping_info.get_mapping_L1_tile_size()[3],
                                                                                            mapping_info.get_mapping_L1_tile_size()[4],
                                                                                            mapping_info.get_mapping_L1_tile_size()[5])
        mapping_data_L2_temp['permutation'] = "N{}{}{}{}{}{}".format(DIM.to_str_timeloop(mapping_info.get_mapping_l1_order()[0]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_l1_order()[1]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_l1_order()[2]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_l1_order()[3]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_l1_order()[4]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_l1_order()[5]))



        #L1 map(=1)
        mapping_data_L1_temp['factors'] = "M=1 C=1 P=1 Q=1 R=1 S=1 N=1"
        mapping_data_L1_temp['permutation'] = "SRQPCMN"

        #Parallel map
        hw_map = [1 for i in range(0,6)]  #dim order
        hw_map[hw_info.get_YDim()] = hw_info.get_Y()
        hw_map[hw_info.get_XDim()] = hw_info.get_X()
        mapping_data_L1_spat['factors'] = "M={} C={} P={} Q={} R={} S={} N=1".format(hw_map[0],hw_map[1],hw_map[2],hw_map[3],hw_map[4],hw_map[5])
        mapping_data_L1_spat['permutation'] = "SRPQCMN"

        mapping_data_Buf['factors'] = "M={} C={} P={} Q={} R={} S={} N=1".format(mapping_info.get_mapping_PE_tile_size()[0],
                                                                                            mapping_info.get_mapping_PE_tile_size()[1],
                                                                                            mapping_info.get_mapping_PE_tile_size()[2],
                                                                                            mapping_info.get_mapping_PE_tile_size()[3],
                                                                                            mapping_info.get_mapping_PE_tile_size()[4],
                                                                                            mapping_info.get_mapping_PE_tile_size()[5])
        #mapping_data_Buf['factors'] = "M=1 C=1 P=1 Q=1 R=1 S=1 N=1"
        
        ##############FIXME!!!!! -> add pe order?
        mapping_data_Buf['permutation'] = "SRQPCMN"
        #mapping_data_Buf['permutation'] = "N{}{}{}{}{}{}".format(DIM.to_str_timeloop(mapping_info.get_mapping_l1_order()[0]),
        #                                                                    DIM.to_str_timeloop(mapping_info.get_mapping_l1_order()[1]),
        #                                                                    DIM.to_str_timeloop(mapping_info.get_mapping_l1_order()[2]),
        #                                                                    DIM.to_str_timeloop(mapping_info.get_mapping_l1_order()[3]),
        #                                                                    DIM.to_str_timeloop(mapping_info.get_mapping_l1_order()[4]),
        #                                                                    DIM.to_str_timeloop(mapping_info.get_mapping_l1_order()[5]))

        with open(file_name, "w") as f:
            yaml.dump(mapping_data, f)
        return file_name

    def write_timeloop_sparseopt(self, dimension, hw_info, mapping_info, filename_=None):
        file_name = self.result_yaml_sparseopt if filename_ is None else filename_
        example_file_name = self.example_yaml_sparseopt
        with open(example_file_name, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        ## compression rank must be equal to Buffer temporal rank
        # get buffer temporal rank = (# of elet > 1) - (# of elt == 1)
        rank_map = [i for i in mapping_info.get_mapping_PE_tile_size()]  #dim order
        rank_map[hw_info.get_XDim()] *= hw_info.get_X()
        rank_map[hw_info.get_YDim()] *= hw_info.get_Y()
        compression_rank = (6 - rank_map.count(1))
        # set compression method
        data_L1 = data['sparse_optimizations']['targets'][0]
        data_L1_inputs = data_L1['representation-format']['data-spaces'][0]
        data_L1_inputs['ranks'] = [{'format': 'UB'} for i in range(compression_rank)]

        with open(file_name, "w") as f:
            yaml.dump(data, f)
        return file_name

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

        process = []
        main_m_file = self.random_file_name
        for i in range(lim):
            # Run timeloop
            m_file = main_m_file+"{}".format(i)
            try:
                hw_info, mapping_info = self.gene2mapping(dimension[i], hw_gene[i], map_gene[i])
            except Exception as e:
                # Invalid mapping
                #print(e)
                return None, (None, None), None
            hw_file_name = self.write_timeloop_hw(dimension[i], hw_info, mapping_info, filename_=None if thread_id is None else self.result_yaml_hw+str(thread_id))
            map_file_name = self.write_timeloop_mapping(dimension[i], hw_info, mapping_info, filename_=None if thread_id is None else self.result_yaml_map+str(thread_id))
            prob_file_name = self.write_timeloop_problem(dimension[i], hw_info, mapping_info, filename_=None if thread_id is None else self.result_yaml_prob+str(thread_id))
            sparse_file_name = self.write_timeloop_sparseopt(dimension[i], hw_info, mapping_info, filename_=None if thread_id is None else self.result_yaml_sparseopt+str(thread_id))
            os.remove(result_dir+"timeloop-model.map+stats.xml") if os.path.exists(result_dir+"timeloop-model.map+stats.xml") else None
            os.makedirs(result_dir) if os.path.exists(result_dir) is False else None
            command = ["../../../timeloop/build/timeloop-model",
                    hw_file_name,
                    prob_file_name,
                    map_file_name,
                    sparse_file_name,
                    "-o", result_dir]
            process.append(Popen(command, stdout=PIPE, stderr=PIPE))
        for i in range(lim):
            stdout, stderr = process[i].communicate()
            if stderr != b'':
                #print("Output: \n", stdout.decode('ascii'))
                #print("Error Code: \n", stderr.decode('ascii'))
                #raise "Timeloop/MAESTRO compile error"
                # Invalid mapping
                return None, (None, None), None
            process[i].wait()

        # Get timeloop result
        try:
            import xml.etree.ElementTree as elemTree
            timeloop_stat = elemTree.parse(result_dir+'timeloop-model.map+stats.xml').getroot()
            timeloop_stat_item_px_macc = timeloop_stat[0][0][0][2][0]
            timeloop_stat_item_px_buffer = timeloop_stat[0][0][0][3][0]
            timeloop_stat_item_px_l1 = timeloop_stat[0][0][0][4][0]
            timeloop_stat_item_px_l2 = timeloop_stat[0][0][0][5][0]
            timeloop_stat_item_px_dram = timeloop_stat[0][0][0][6][0]
            if timeloop_stat_item_px_buffer[2][0][0].text!='Buffer' or timeloop_stat_item_px_l1[2][0][0].text!='L1' \
                     or timeloop_stat_item_px_l2[2][0][0].text!='L2' or timeloop_stat_item_px_dram[2][0][0].text!='DRAM':
                print("timeloop output error ???")
                raise "timeloop output error"
            f = open(result_dir+"timeloop-model.stats.txt", "r")
            while True:
                line = f.readline()
                if not line: break
                if "Summary Stats" in line: break
                # print(line)
            f.readline()  # ----

            #Get metrics(observation)
            buffer_weight = timeloop_stat_item_px_buffer[3][2][0][0].text
            buffer_input = timeloop_stat_item_px_buffer[3][2][0][1].text
            buffer_output = timeloop_stat_item_px_buffer[3][2][0][2].text
            l1_weight = timeloop_stat_item_px_l1[3][2][0][0].text
            l1_input = timeloop_stat_item_px_l1[3][2][0][1].text
            l1_output = timeloop_stat_item_px_l1[3][2][0][2].text
            l2_weight = timeloop_stat_item_px_l2[3][2][0][0].text
            l2_input = timeloop_stat_item_px_l2[3][2][0][1].text
            l2_output = timeloop_stat_item_px_l2[3][2][0][2].text
            util = float(f.readline().split(' ')[1])  # utilization
            runtime = float(f.readline().split(' ')[1])  # cycles
            energy = float(f.readline().split(' ')[1])  # energy
            area = float(f.readline().split(' ')[1])  # area
            l2_size = l2_weight + l2_input + l2_output
            l1_size = l1_weight + l1_input + l1_output
            mac = 1 #unsupport
            power = 1 #unsupport
            observation = [runtime, 1, energy, area, l1_size, l2_size, mac, power]
            estimated = {'util': util, 'cycle': runtime, 'energy': energy, 'area': area, 'mac': mac, 'power': power,
                        'buffer_weight': buffer_weight, 'buffer_input': buffer_input, 'buffer_output': buffer_output,
                        'l1_weight': l1_weight, 'l1_input': l1_input, 'l1_output': l1_output,
                        'l2_weight': l2_weight, 'l2_input': l2_input, 'l2_output': l2_output }

            os.remove(result_dir+"timeloop-model.map+stats.xml")  if os.path.exists(result_dir+"timeloop-model.map+stats.xml") else None
            os.remove(result_dir+"timeloop-model.map.txt")  if os.path.exists(result_dir+"timeloop-model.map.txt") else None
            os.remove(result_dir+"timeloop-model.stats.txt")  if os.path.exists(result_dir+"timeloop-model.stats.txt") else None
            hw_info, mapping_info = self.gene2mapping(dimension[0], hw_gene[0], map_gene[0]) #lim must be 1
            target_constraint.set_constraint(group_density=hw_info.get_group_density(), bank=hw_info.get_bank())
            if runtime <= 0 or energy <= 0: #Invalid
                raise Exception('invalid map: runtime')
            if (target_constraint is not None) and (target_constraint.check_constraints(estimated, hw_info, mapping_info) is False): #Check constraint Invalid
                '''
                if gen>=10: # ignore resource check before fore generation
                    #print("Gene has out of FPGA resource, then raise error")
                    raise Exception('invalid map: constraint')
                else:
                    #print("Initial adv. although Gene has out of FPGA resource")
                    pass
                '''
                raise Exception('invalid map: constraint')
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
            hw_file_name = self.write_timeloop_hw(dimension[i], hw_info, mapping_info, filename_=hw_file_name)
            map_file_name = self.write_timeloop_mapping(dimension[i], hw_info, mapping_info, filename_=map_file_name)
            prob_file_name = self.write_timeloop_problem(dimension[i], hw_info, mapping_info, filename_=prob_file_name)
            sparse_file_name = self.write_timeloop_sparseopt(dimension[i], hw_info, mapping_info, filename_=sparse_file_name)

    def get_gene_HW_info(self, dimension, hw_gene, map_gene):
        hw_info, mapping_info = self.gene2mapping(dimension, hw_gene, map_gene)
        return [DIM.to_str(hw_info.get_XDim()), hw_info.get_X(), \
                DIM.to_str(hw_info.get_YDim()), hw_info.get_Y(), \
                hw_info.get_group_density(), hw_info.get_bank()]

