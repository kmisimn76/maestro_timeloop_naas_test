#from enum import Enum
from enum import IntEnum
class HW_GENE (IntEnum):
    L2_SIZE     = 0
    L1_SIZE     = 1
    NUM_PE      = 2
    BW          = 3
    NUM_DIM     = 4
    DIM_SIZE_0  = 5
    DIM_SIZE_1  = 6
    DIM_SIZE_2  = 7
    PAR_DIM_K   = 8
    PAR_DIM_C   = 9
    PAR_DIM_Y   = 10
    PAR_DIM_X   = 11
    PAR_DIM_R   = 12
    PAR_DIM_S   = 13
class MAPPING_GENE (IntEnum):
    ARR_LOOP_ORDER_K = 0
    ARR_LOOP_ORDER_C = 1
    ARR_LOOP_ORDER_Y = 2
    ARR_LOOP_ORDER_X = 3
    ARR_LOOP_ORDER_R = 4
    ARR_LOOP_ORDER_S = 5
    ARR_TILE_SIZE_K  = 6
    ARR_TILE_SIZE_C  = 7
    ARR_TILE_SIZE_Y  = 8
    ARR_TILE_SIZE_X  = 9
    ARR_TILE_SIZE_R  = 10
    ARR_TILE_SIZE_S  = 11
    PE_LOOP_ORDER_K  = 12
    PE_LOOP_ORDER_C  = 13
    PE_LOOP_ORDER_Y  = 14
    PE_LOOP_ORDER_X  = 15
    PE_LOOP_ORDER_R  = 16
    PE_LOOP_ORDER_S  = 17
class DIM (IntEnum):
    K = 0
    C = 1
    Y = 2
    X = 3
    R = 4
    S = 5
    P = 6
    def to_str(data):
        if data==DIM.K: return "K"
        if data==DIM.C: return "C"
        if data==DIM.Y: return "Y"
        if data==DIM.X: return "X"
        if data==DIM.R: return "R"
        if data==DIM.S: return "S"
        if data==DIM.P: return "P"
        #print(data)
        raise "Exception"
    def to_str_timeloop(data):
        if data==DIM.K: return "M"
        if data==DIM.C: return "C"
        if data==DIM.Y: return "Q"
        if data==DIM.X: return "P"
        if data==DIM.R: return "R"
        if data==DIM.S: return "S"
        if data==DIM.P: return "Z"
        #print(data)
        raise "Exception"
class TIMELOOP_HW:
    def __init__(self):
        self.X = 0
        self.Y = 0
        self.XDim = DIM.X
        self.YDim = DIM.Y
        self.L2_size = 0
        self.L1_size = 0
    def set_HW(self,x,y,xdim, ydim, l2_size,l1_size):
        self.X = x
        self.Y = y
        self.XDim = xdim
        self.YDim = ydim
        self.L2_size = l2_size
        self.L1_size = l1_size
    def get_X(self):
        return self.X
    def get_Y(self):
        return self.Y
    def get_XDim(self):
        return self.XDim
    def get_YDim(self):
        return self.YDim
    def get_L2_size(self):
        return self.L2_size
    def get_L1_size(self):
        return self.L1_size
class TIMELOOP_MAPPING:
    def __init__(self):
        self.mapping_gene_raw = None
        self.l_info = None # layer information : K C Y X R S
        self.mapping_tile_size = None
        self.mapping_array_order = None
        self.mapping_pe_order = None
    def set_mapping_gene(self, l_info, dim_sz, hw_gene, gene):
        self.mapping_gene_raw = gene
        self.l_info = l_info
        self.mapping_tile_size = [int(self.mapping_gene_raw[i]*self.l_info[j])+1 for (i, j) in zip(range(MAPPING_GENE.ARR_TILE_SIZE_K, MAPPING_GENE.ARR_TILE_SIZE_S+1), range(0,6))]
        self.dim_size = dim_sz
        selected_hw_dim = sorted(list(enumerate(hw_gene[HW_GENE.PAR_DIM_K:HW_GENE.PAR_DIM_S+1])), key=lambda x:x[1])[-int(hw_gene[HW_GENE.NUM_DIM]):]
        sorted_arr_map_dim = sorted(list(enumerate(self.mapping_gene_raw[MAPPING_GENE.ARR_LOOP_ORDER_K:MAPPING_GENE.ARR_LOOP_ORDER_S+1])), key=lambda x:x[1])
        sorted_pe_map_dim = sorted(list(enumerate(self.mapping_gene_raw[MAPPING_GENE.PE_LOOP_ORDER_K:MAPPING_GENE.PE_LOOP_ORDER_S+1])), key=lambda x:x[1])
        self.mapping_array_order = [sorted_arr_map_dim[i][0] for i in range(0,6)]
        self.mapping_pe_order = [sorted_pe_map_dim[i][0] for i in range(0,6)]
        self.mapping_selected_hw_dim = [selected_hw_dim[0][0], selected_hw_dim[1][0]]
    def get_mapping_L2_tile_size(self):
        output = [int(self.l_info[i]/self.mapping_tile_size[i])+1 for i in range(0,6)]
        output[self.mapping_selected_hw_dim[0]] = 1
        output[self.mapping_selected_hw_dim[1]] = 1
        return output
    def get_mapping_L1_tile_size(self):
        output = [self.mapping_tile_size[i] for i in range(0,6)]
        output[self.mapping_selected_hw_dim[0]] = int(self.l_info[self.mapping_selected_hw_dim[0]]/self.dim_size[0]) +1
        output[self.mapping_selected_hw_dim[1]] = int(self.l_info[self.mapping_selected_hw_dim[1]]/self.dim_size[1]) +1
        return output
    def get_mapping_parallel_size(self):
        output = [1 for i in range(0,6)]
        output[self.mapping_selected_hw_dim[0]] = int(self.dim_size[0])
        output[self.mapping_selected_hw_dim[1]] = int(self.dim_size[1])
        return output
    def get_mapping_array_order(self):
        return [self.mapping_array_order[i] for i in range(0,6)]
    def get_mapping_pe_order(self):
        return [self.mapping_pe_order[i] for i in range(0,6)]

