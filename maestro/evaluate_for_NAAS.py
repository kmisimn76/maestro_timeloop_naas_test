import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools.jupyter_notebook.graph_util import *
import datetime
from tqdm import tqdm
import subprocess

start = datetime.datetime.now()
total_power = 0
for i in tqdm(range(1000)):
    os.system("./maestro --HW_file='data/hw/accelerator_1.m' --Mapping_file='data/mapping/Resnet50_kcp_ws.m' --print_res=false --print_res_csv_file=true --print_log_file=false")
#    cmd = ["./maestro", "--HW_file='data/hw/accelerator_1.m'", "--Mapping_file='data/mapping/Resnet50_kcp_ws.m'", "--print_res=false", "--print_res_csv_file=true", "--print_log_file=false"]
#    subprocess.call(' '.join(cmd), shell=True)

    resnet50_kcp_ws_data = pd.read_csv('./Resnet50_kcp_ws.csv')

#print(resnet50_kcp_ws_data.columns)
# Index(['Neural Network Name', ' Layer Number', ' NumPEs', ' Runtime (Cycles)',
#       ' Activity count-based Energy (nJ)', ' Throughput (MACs/Cycle)',
#       ' Throughput Per Energy (GMACs/s*J)', ' Area', ' Power',
#       ' NoC BW Req (Elements/cycle)', ' Avg BW Req', ' Peak BW Req',
#       ' Vector Width', ' Offchip BW Req (Elements/cycle)',
#       '  L2 SRAM Size Req (Bytes)', ' L1 SRAM Size Req (Bytes)',
#       ' Multicasting Factor (Weight)', ' Multicasting Factor (Input)',
#       ' Num Total Input Pixels', ' Num Total Weight Pixels', ' Ops/J',
#       ' Num MACs', ' PE power', ' L1 power', ' L2 power', ' NOC power',
#       ' input l1 read', ' input l1 write', ' input l2 read',
#       ' input l2 write', ' input reuse factor', 'filter l1 read',
#       ' filter l1 write', ' filter l2 read', ' filter l2 write',
#       ' filter reuse factor', 'output l1 read', ' output l1 write',
#       ' output l2 read', ' output l2 write', ' output reuse factor',
#       'Ingress Delay (Min)', ' Ingress Delay (Max)', ' Ingress Delay (Avg)',
#       ' Egress Delay (Min)', ' Egress Delay (Max)', '  Egress Delay (Avg)',
#       'Compute Delay (Min)', ' Compute Delay (Min)', ' Compute Delay (Avg)',
#       'Avg number of utilized PEs', ' Arithmetic Intensity'],
#      dtype='object')
#print(resnet50_kcp_ws_data.head())
    total_power += resnet50_kcp_ws_data[' Activity count-based Energy (nJ)']
    #print(total_power)

print(total_power)
end = datetime.datetime.now()
c = end - start
print(c)
