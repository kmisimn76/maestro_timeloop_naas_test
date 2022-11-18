import sys
print("usage: python print_mapping.py ./res")
file_path = sys.argv[1] if len(sys.argv)>1 else "res"

for iteration in range(53):
    with open("{}/{:02d}/hw_.yaml".format(file_path, iteration), "r") as f:
        data = []
        for line in f:
            data.append(line[:-1].split(','))
        hw_data = data[0]
        data = data[1:6]
    
        # appendix data
        for i in range(len(data)): # n index
            data[i] = data[i]+['1']
        conv_layer = [str(int(float(data[0][i]))*int(float(data[2][i]))*int(float(data[4][i]))) for i in range(6)]
        sparsity = "0.99"
    
        out_str = ""
        for line in data:
            line = [str(int(float(l))) for l in line]
            out_str += ','.join(line)+":"
        out_str = "{}:{}:{}".format(','.join(conv_layer), sparsity, out_str)
        print(out_str[:-1])
print(int(float(hw_data[0])), int(float(hw_data[1])), int(float(hw_data[2])), int(float(hw_data[3])))
print(int(float(hw_data[4])), int(float(hw_data[5])))
if int(float(hw_data[1]))==1: #output stationary
    bk = int(float(hw_data[5]))
    bank_wt  = bk * int(float(hw_data[0])) #wt -> K
    bank_in  = bk * int(float(hw_data[2])) if int(float(hw_data[2])) != 1 else int(float(hw_data[3])) #in -> W
    bank_out = bk * int(float(hw_data[2])) if int(float(hw_data[2])) != 1 else int(float(hw_data[3])) #out-> W
elif int(float(hw_data[2]))==1 and int(float(hw_data[3]))==1: #wt stationary
    bk = int(float(hw_data[5]))
    bank_wt  = int(float(hw_data[0])) #wt -> C
    bank_in  = bk * int(float(hw_data[1])) #in -> C
    bank_out = bk * int(float(hw_data[0])) #out-> K
import math
print(int(math.ceil(float(hw_data[6]))), int(math.ceil(float(hw_data[7]))), int(math.ceil(float(hw_data[8]))))
print("(", int(math.ceil(math.ceil(float(hw_data[6]))/bank_wt)), int(math.ceil(math.ceil(float(hw_data[7]))/bank_in)), int(math.ceil(math.ceil(float(hw_data[8]))/bank_out)),")")
print(int(math.ceil(float(hw_data[9]))), int(math.ceil(float(hw_data[10]))), int(math.ceil(float(hw_data[11]))))
print("(", int(math.ceil(math.ceil(float(hw_data[9]))/bank_wt)), int(math.ceil(math.ceil(float(hw_data[10]))/bank_in)), int(math.ceil(math.ceil(float(hw_data[11]))/bank_out)),")")
print("sparse" if hw_data[12]=="True" else "dense")
