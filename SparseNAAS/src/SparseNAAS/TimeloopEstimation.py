
from subprocess import Popen, PIPE
import yaml
import numpy as np
import os

from TimeloopMapping import *

df_dict = {1:"dla", 2:"shi", 3:"eye"}
m_type_dicts = {1:"CONV", 2:"DSCONV"}

############## ----------------------- ###################
############## ----------------------- ###################
# Timeloop Perform Estimation
############## ----------------------- ###################
############## ----------------------- ###################

class TimeloopEstimator():

    def __init__(self, random_file_name):
        self.random_file_name = random_file_name
        self.example_yaml_hw = "timeloop_example_yaml/Example_timeloop_hw.yaml"
        self.example_yaml_prob = "timeloop_example_yaml/conv1d.prob.yaml"
        self.result_yaml_hw = "../../data/timeloop/hw_.yaml"
        self.result_yaml_prob = "../../data/timeloop/problem_.yaml"
        self.result_yaml_map = "../../data/timeloop/mapping_.yaml"

    def softmax(self, a) :
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def gene2mapping(self, dimension, hw_gene, map_gene):
        selected_hw_dim = sorted(list(enumerate(hw_gene[HW_GENE.PAR_DIM_K:HW_GENE.PAR_DIM_S+1])), key=lambda x:x[1])[-int(hw_gene[HW_GENE.NUM_DIM]):]
        dim_size = [int(hw_gene[HW_GENE.NUM_PE]**x) for x in self.softmax([hw_gene[HW_GENE.DIM_SIZE_0], hw_gene[HW_GENE.DIM_SIZE_1]])] #get dim size from encoding vector
        hw_info = TIMELOOP_HW()
        mapping = TIMELOOP_MAPPING()
        hw_info.set_HW(dim_size[0], dim_size[1], selected_hw_dim[0][0], selected_hw_dim[1][0], hw_gene[HW_GENE.L2_SIZE], hw_gene[HW_GENE.L1_SIZE])
        mapping.set_mapping_gene(dimension[0:6], dim_size, hw_gene, map_gene)
        return hw_info, mapping

    def write_timeloop_hw(self, dimension, hw_info, m_file=None, layer_id=0):
        file_name = self.result_yaml_hw
        example_file_name = self.example_yaml_hw
        with open(example_file_name, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        x_size = hw_info.get_X()
        y_size = hw_info.get_Y()
        data['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['name'] = 'PE[0..{}]'.format(max(x_size*y_size-1,1)) # Exception case(incur Timeloop error); when x=y=1
        data['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local'][0]['attributes']['meshX'] = x_size
        data['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local'][1]['attributes']['meshX'] = x_size
        with open(file_name, "w") as f:
            yaml.dump(data, f)
        return file_name

    def write_timeloop_problem(self, dimension, mapping_info, m_file=None, layer_id=0):
        file_name = self.result_yaml_prob
        example_file_name = self.example_yaml_prob
        with open(example_file_name, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        l2_size_ = mapping_info.get_mapping_L2_tile_size()
        l1_size_ = mapping_info.get_mapping_L1_tile_size()
        par_size_ = mapping_info.get_mapping_parallel_size()
        dim_size_ = [i*j*l for (i,j,l) in zip(l2_size_, l1_size_, par_size_)]
        for i in range(0,6):
            if dim_size_[i] < dimension[i]:
                print("Dim size must be large than problem dimension: ", dim_size_[i], dimension[i])
                raise 'Mapping Err'
        data['problem']['instance']['M'] = int(dim_size_[0])
        data['problem']['instance']['C'] = int(dim_size_[1])
        data['problem']['instance']['P'] = int(dim_size_[2])
        data['problem']['instance']['Q'] = int(dim_size_[3])
        data['problem']['instance']['R'] = int(dim_size_[4])
        data['problem']['instance']['S'] = int(dim_size_[5])
        with open(file_name, "w") as f:
            yaml.dump(data, f)
        return file_name

    def write_timeloop_mapping(self, dimension, hw_info, mapping_info, m_file=None, layer_id=0):
        if len(dimension) > 6:
            m_type = m_type_dicts[int(dimension[-1])]
        else:
            m_type = "CONV"
        # Dataflow description
        file_name = self.result_yaml_map

        mapping_data = {"mapping":
                            [
                                {"target":"DRAM",
                                    "type":"temporal",
                                    "factors":"C=1 M=1 R=1 S=1 N=1 P=1 Q=1", "permutation":"SRQPCMN"},
                                {"target":"L2_spad",
                                    "type":"temporal",
                                    "factors":"", "permutation":""},
                                {"target":"L2_spad",
                                    "type":"spatial",
                                    "factors":"", "permutation":""},
                                {"target":"Buffer",
                                    "type":"temporal",
                                    "factors":"", "permutation":""}
                            ]
                        }
        mapping_data['mapping'][1]['factors'] = "M={} C={} P={} Q={} R={} S={} N=1".format(mapping_info.get_mapping_L2_tile_size()[0],
                                                                                            mapping_info.get_mapping_L2_tile_size()[1],
                                                                                            mapping_info.get_mapping_L2_tile_size()[2],
                                                                                            mapping_info.get_mapping_L2_tile_size()[3],
                                                                                            mapping_info.get_mapping_L2_tile_size()[4],
                                                                                            mapping_info.get_mapping_L2_tile_size()[5])
        mapping_data['mapping'][1]['permutation'] = "N{}{}{}{}{}{}".format(DIM.to_str_timeloop(mapping_info.get_mapping_array_order()[0]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_array_order()[1]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_array_order()[2]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_array_order()[3]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_array_order()[4]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_array_order()[5]))
        hw_map = [1 for i in range(0,6)]  #dim order
        hw_map[hw_info.get_XDim()] = hw_info.get_X()
        hw_map[hw_info.get_YDim()] = hw_info.get_Y()
        mapping_data['mapping'][2]['factors'] = "M={} C={} P={} Q={} R={} S={} N=1".format(hw_map[0],hw_map[1],hw_map[2],hw_map[3],hw_map[4],hw_map[5])
        mapping_data['mapping'][2]['permutation'] = "N{}{}{}{}{}{}".format(DIM.to_str_timeloop(mapping_info.get_mapping_pe_order()[0]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_pe_order()[1]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_pe_order()[2]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_pe_order()[3]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_pe_order()[4]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_pe_order()[5]))
        mapping_data['mapping'][3]['factors'] = "M={} C={} P={} Q={} R={} S={} N=1".format(mapping_info.get_mapping_L1_tile_size()[0],
                                                                                            mapping_info.get_mapping_L1_tile_size()[1],
                                                                                            mapping_info.get_mapping_L1_tile_size()[2],
                                                                                            mapping_info.get_mapping_L1_tile_size()[3],
                                                                                            mapping_info.get_mapping_L1_tile_size()[4],
                                                                                            mapping_info.get_mapping_L1_tile_size()[5])
        mapping_data['mapping'][3]['permutation'] = "N{}{}{}{}{}{}".format(DIM.to_str_timeloop(mapping_info.get_mapping_pe_order()[0]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_pe_order()[1]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_pe_order()[2]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_pe_order()[3]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_pe_order()[4]),
                                                                            DIM.to_str_timeloop(mapping_info.get_mapping_pe_order()[5]))

        with open(file_name, "w") as f:
            yaml.dump(mapping_data, f)
        return file_name

    def observe_timeloop(self, dimension, hw_gene, map_gene, judge, firsttime=False, multi=False):
        if multi==True:
            if (len(dimension) != len(hw_gene)) or (len(dimension) != len(map_gene)):
                raise "Invalid Argument"
            lim = len(dimension)
        else:
            lim = 1
            dimension = [dimension]
            hw_gene = [hw_gene]
            map_gene = [map_gene]
        process = []
        main_m_file = self.random_file_name
        for i in range(lim):
            m_file = main_m_file+"{}".format(i)
            hw_info, mapping_info = self.gene2mapping(dimension[i], hw_gene[i], map_gene[i])
            hw_file_name = self.write_timeloop_hw(dimension[i], hw_info)
            map_file_name = self.write_timeloop_mapping(dimension[i], hw_info, mapping_info)
            prob_file_name = self.write_timeloop_problem(dimension[i], mapping_info)
            #HW gene
            os.remove("./timeloop-model.map+stats.xml") if os.path.exists("./timeloop-model.map+stats.xml") else None
            command = ["../../../timeloop/build/timeloop-model",
                    hw_file_name,
                    prob_file_name,
                    map_file_name]
            process.append(Popen(command, stdout=PIPE, stderr=PIPE))
        for i in range(lim):
            stdout, stderr = process[i].communicate()
            if stderr != b'':
                print("Output: \n", stdout.decode('ascii'))
                print("Error Code: \n", stderr.decode('ascii'))
                raise "Timeloop/MAESTRO compile error"
            process[i].wait()

        #print(command, stdout, self.gene2dataflow(dimension, hw_gene, map_gene).mapping)
        import xml.etree.ElementTree as elemTree
        try:
            f = open("./timeloop-model.stats.txt", "r")
            while True:
                line = f.readline()
                if not line: break
                if "Summary Stats" in line: break
                # print(line)
            f.readline()  # ----
            util = float(f.readline().split(' ')[1])  # utilization
            runtime = float(f.readline().split(' ')[1])  # cycles
            energy = float(f.readline().split(' ')[1])  # energy
            area = float(f.readline().split(' ')[1])  # area
            #print(runtime, energy)
            os.remove("./timeloop-model.map+stats.xml")  if os.path.exists("./timeloop-model.map+stats.xml") else None
            os.remove("./timeloop-model.map.txt")  if os.path.exists("./timeloop-model.map.txt") else None
            os.remove("./timeloop-model.stats.txt")  if os.path.exists("./timeloop-model.stats.txt") else None
            if runtime <= 0 or energy <= 0: #or throughput[0][0] <= 0:
                #print(runtime[0][0])
                #print(energy[0][0])
                #print(throughput[0][0])
                raise
            #observation = [runtime, 1, energy, area, l1_size, l2_size, mac, power]
            observation = [runtime, 1, energy, area, 1, 1, 1, 1]
            #print("pass")
            return observation, judge(observation)
        except:
            raise "compile err?"
            return None, None
