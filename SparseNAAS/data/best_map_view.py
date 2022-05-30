
## Warning
print("for KxQ output statoinary!")

import yaml
for i in range(0,53):
    result_yaml_map = "./best/{0:02d}/mapping_.yaml".format(i)
    file_name = result_yaml_map
    with open(file_name, "r") as f:
        mapping_data = yaml.load(f, Loader=yaml.FullLoader)
    
    mapping_data_DRAM_temp = mapping_data['mapping'][0]
    mapping_data_L2_temp = mapping_data['mapping'][1]
    mapping_data_L1_temp = mapping_data['mapping'][2]
    mapping_data_L1_spat = mapping_data['mapping'][3]
    mapping_data_Buf = mapping_data['mapping'][4]
    
    l2_tilenum = [int(d[2:]) for d in mapping_data_DRAM_temp['factors'].split(' ')]
    l1_tilenum = [int(d[2:]) for d in mapping_data_L2_temp['factors'].split(' ')]
    tilesize = [int(d[2:]) for d in mapping_data_L1_temp['factors'].split(' ')]
    tilesize = [t*int(d[2:]) for t,d in zip(tilesize,mapping_data_L1_spat['factors'].split(' '))]
    tilesize = [t*int(d[2:]) for t,d in zip(tilesize,mapping_data_Buf['factors'].split(' '))]
    
    # for KxQ output stationary
    l1_tilenum[3] *= tilesize[3]
    tilesize[3] = 1
    
    print(*l2_tilenum, *l1_tilenum, *tilesize, sep=',')
