
for iteration in range(53):
    with open("res/{:02d}/hw_.yaml".format(iteration), "r") as f:
        data = []
        for line in f:
            data.append(line[:-1].split(','))
        data = data[1:6]
    
        # appendix data
        for i in range(len(data)): # n index
            data[i] = data[i]+['1']
        conv_layer = [str(int(data[0][i])*int(data[2][i])*int(data[4][i])) for i in range(6)]
        sparsity = "0.99"
    
        out_str = ""
        for line in data:
            out_str += ','.join(line)+":"
        out_str = "{}:{}:{}".format(','.join(conv_layer), sparsity, out_str)
        print(out_str[:-1])
