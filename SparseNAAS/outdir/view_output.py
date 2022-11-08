import pickle
import numpy
import sys

file_path = sys.argv[1] #ex) outdir/res/resnet50.../result_c.plt
f = open(file_path, "rb")
data = pickle.load(f)
print(data)
