#!/bin/bash
#../../build/timeloop-model ./arch/1level_sparse.arch.yaml ./comp/metadata.yaml ./map/conv1d-1level-sparse.map.yaml ./sparse_opt/sparse.yaml ./prob/conv1d_sparse.prob.yaml -o output_sparse_only/
../../build/timeloop-model ./arch/arch_.yaml ./comp/metadata.yaml ./map/map_.yaml ./sparse_opt/sparse_.yaml ./prob/prob_.yaml -o output_sparse/
grep "Cycles: " output_sparse/timeloop-model.stats.txt
#tail -n 30 output_sparse/timeloop-model.stats.txt

#grep "Cycles: " output_vgg16/*/timeloop-model.stats.txt
