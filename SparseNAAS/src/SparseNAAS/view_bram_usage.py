import sys
file_path = sys.argv[1] if len(sys.argv)>1 else "res"

print("usage: python view_buffer.py ../../outdir/res")
with open("{}/{:02d}/hw_.yaml".format(file_path, 0), "r") as f:
    data = []
    for line in f:
        data.append(line[:-1].split(','))
    hw_data = data[0]
print(int(float(hw_data[0])), int(float(hw_data[1])), int(float(hw_data[2])), int(float(hw_data[3])))
print(int(float(hw_data[4])), int(float(hw_data[5])))
if int(float(hw_data[1]))==1: #output stationary
    bk = int(float(hw_data[5]))
    bank_wt  = bk * int(float(hw_data[0])) #wt -> K
    bank_in  = bk * int(float(hw_data[2])) if int(float(hw_data[2])) != 1 else int(float(hw_data[3])) #in -> W
    bank_out = bk * int(float(hw_data[2])) if int(float(hw_data[2])) != 1 else int(float(hw_data[3])) #out-> W
elif int(float(hw_data[2]))==1 and int(float(hw_data[3]))==1: #wt stationary
    bk = int(float(hw_data[5]))
    bank_wt  = int(float(hw_data[1])) #wt -> C
    bank_in  = bk * int(float(hw_data[1])) #in -> C
    bank_out = bk * int(float(hw_data[0])) #out-> K
import math
print(int(math.ceil(float(hw_data[6]))), int(math.ceil(float(hw_data[7]))), int(math.ceil(float(hw_data[8]))))
print("(", int(math.ceil(math.ceil(float(hw_data[6]))/bank_wt)), int(math.ceil(math.ceil(float(hw_data[7]))/bank_in)), int(math.ceil(math.ceil(float(hw_data[8]))/bank_out)),")")
print(int(math.ceil(float(hw_data[9]))), int(math.ceil(float(hw_data[10]))), int(math.ceil(float(hw_data[11]))))
print("(", int(math.ceil(math.ceil(float(hw_data[9]))/bank_wt)), int(math.ceil(math.ceil(float(hw_data[10]))/bank_in)), int(math.ceil(math.ceil(float(hw_data[11]))/bank_out)),")")
print("sparse" if hw_data[12]=="True" else "dense")


from FPGAConstraint.AlveoU200 import *
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
    l2_weight = int(float(hw_data[6]))
    l2_input = int(float(hw_data[7]))
    l2_output = int(float(hw_data[8]))
    l1_weight = int(float(hw_data[9]))
    l1_input = int(float(hw_data[10]))
    l1_output = int(float(hw_data[11]))
    if int(float(hw_data[1]))==1: #out stationary
        PEX=int(float(hw_data[0]))
        PEY=int(float(hw_data[2])) if int(float(hw_data[2])) != 1 else int(float(hw_data[3]))
        weight_stationary = False
        output_stationary = True
        input_stationary = False
        density = int(float(hw_data[4]))
    elif int(float(hw_data[2]))==1 and int(float(hw_data[3]))==1: #wt stationary
        PEX=int(float(hw_data[0]))
        PEY=int(float(hw_data[1]))
        weight_stationary = True 
        output_stationary = False
        input_stationary = False
        density = int(float(hw_data[4]))
    st_order = [5,6,4,3,2,1] if weight_stationary else ([5,4,6,3,2,1] if output_stationary else [4,5,6,3,2,1])
    density = int(math.log(density, 2))
    bank = int(float(hw_data[5]))
    use_sparsity = True if hw_data[12]=="True" else False
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
    constraint.set_constraint(group_density=2**density, bank=bank)
    print("HW info")
    print(PEX,PEY,2**density, bank)
    print(hw_info.get_X(), hw_info.get_Y())
    print(hw_info.get_XDim(), hw_info.get_YDim())
    print(constraint.check_constraints(estimated, hw_info, mapping_info, use_sparsity, log_=True))
