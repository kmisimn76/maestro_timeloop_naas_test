import os
from subprocess import Popen, PIPE


prob_info = []
import csv
with open('prob/resnet50_sparse.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        prob_info.append(row)
        print(row)

import yaml
def save_prob(result_yaml_prob, example_yaml_prob, layer_info):
    file_name = result_yaml_prob
    example_file_name = example_yaml_prob
    with open(example_file_name, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    C = int(layer_info[1])
    tile_c = 16 if C/16>=16 else (8 if C/16>=8 else (4 if C/16>=4 else (2 if C/16>=2 else 1)))
    C = tile_c*16 if C<tile_c*16 else C
    data['problem']['instance']['M'] = int(layer_info[0])
    data['problem']['instance']['C'] = C #int(layer_info[1])
    data['problem']['instance']['P'] = int(layer_info[2])
    data['problem']['instance']['Q'] = int(layer_info[3])
    data['problem']['instance']['R'] = int(layer_info[4])
    data['problem']['instance']['S'] = int(layer_info[5])
    #data['problem']['instance']['densities']['Inputs']['density'] = layer_info[7] # grp2 density
    data['problem']['instance']['densities']['Inputs']['density'] = layer_info[8] # grp4 density

    with open(file_name, "w") as f:
        yaml.dump(data, f)
    return file_name

def save_map(result_yaml_map, example_yaml_map, layer_info):
    file_name = result_yaml_map
    example_file_name = example_yaml_map
    with open(example_file_name, "r") as f:
        mapping_data = yaml.load(f, Loader=yaml.FullLoader)
    mapping_data_DRAM_temp = mapping_data['mapping'][0]
    mapping_data_L2_temp = mapping_data['mapping'][1]
    mapping_data_L1_temp = mapping_data['mapping'][2]
    mapping_data_L1_spat = mapping_data['mapping'][3]
    mapping_data_Buf = mapping_data['mapping'][4]
    

    K = layer_info[0]
    C = layer_info[1]
    tile_c = 16 if C/16>=16 else (8 if C/16>=8 else (4 if C/16>=4 else (2 if C/16>=2 else 1)))
    C = tile_c*16 if C<tile_c*16 else C
    H = layer_info[2]
    W = layer_info[3]
    R = layer_info[4]
    S = layer_info[5]

    ARRAY_K = 16 #constant
    ARRAY_C = 16 #constant

    TILESIZE_W = 14 if W>=14 else 7
    TILESIZE_H = 14 if H>=14 else 7

    #L1_TILENUM_K = 16 if K/16>=16 else (8 if K/16>=8 else (4 if K/16>=4 else (2 if K/16>=2 else 1)))
    #L1_TILENUM_C = 16 if C/16>=16 else (8 if C/16>=8 else (4 if C/16>=4 else (2 if C/16>=2 else 1)))
    #L1_TILENUM_H = 8 if H/TILESIZE_H>=8 else (4 if H/TILESIZE_H>=4 else (2 if H/TILESIZE_H>=2 else 1))
    #L1_TILENUM_W = 8 if W/TILESIZE_W>=8 else (4 if W/TILESIZE_W>=4 else (2 if W/TILESIZE_W>=2 else 1))
    L1_TILENUM_K = 4 if K/16>=4 else (2 if K/16>=2 else 1)
    L1_TILENUM_C = 4 if C/16>=4 else (2 if C/16>=2 else 1)
    L1_TILENUM_H = 1
    L1_TILENUM_W = 1
    L1_TILENUM_R = 3 if R==3 else 1
    L1_TILENUM_S = 3 if S==3 else 1

    L2_TILENUM_K = int(K/(ARRAY_K*L1_TILENUM_K))
    L2_TILENUM_C = int(C/(ARRAY_C*L1_TILENUM_C))
    L2_TILENUM_H = int(H/(TILESIZE_H*L1_TILENUM_H))
    L2_TILENUM_W = int(W/(TILESIZE_W*L1_TILENUM_W))
    L2_TILENUM_R = int(R/(L1_TILENUM_R))
    L2_TILENUM_S = int(S/(L1_TILENUM_S))

    #DRAM map
    mapping_data_DRAM_temp['factors'] = "M={} C={} P={} Q={} R={} S={} N=1".format(L2_TILENUM_K, L2_TILENUM_C, L2_TILENUM_H, L2_TILENUM_W, L2_TILENUM_R, L2_TILENUM_S)
    #L2 map(=1)
    mapping_data_L2_temp['factors'] = "M={} C={} P={} Q={} R={} S={} N=1".format(L1_TILENUM_K, L1_TILENUM_C, L1_TILENUM_H, L1_TILENUM_W, L1_TILENUM_R, L1_TILENUM_S)

    mapping_data_Buf['factors'] = "M=1 C=1 P={} Q={} R=1 S=1 N=1".format(TILESIZE_H, TILESIZE_W)
    with open(file_name, "w") as f:
        yaml.dump(mapping_data, f)
    return file_name

command = ["rm", "-rf", "output_resnet50"]
process = Popen(command)
process.wait()
command = ["mkdir", "output_resnet50"]
process = Popen(command)
process.wait()


for i in range(1,54):
    l_info = [float(n) for n in prob_info[i]]
    print(l_info)
    # Generate prob_(i)
    save_prob('prob/prob_.yaml', 'prob/conv1d_sparse.prob.yaml', l_info)
    # Generate map_(i)
    save_map('map/map_.yaml', 'map/sparse_npu.map.yaml', l_info)
    command = ["./run_sparse.sh"]
    process = Popen(command)
    process.wait()
    print(command, i)
    command = ["cp", "-r", "output_sparse", "output_resnet50/{:02d}".format(i-1)]
    process = Popen(command)
    process.wait()
