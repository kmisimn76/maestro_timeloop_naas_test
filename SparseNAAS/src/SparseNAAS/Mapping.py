#from TimeloopMapping import *
from HWGene import *
from MapGene import *

def softmax(a) :
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def gene2mapping(dimension, hw_gene, map_gene):


