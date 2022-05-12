../../build/timeloop-model ./arch/1level_sparse.arch.yaml ./comp/metadata.yaml ./map/conv1d-1level-sparse.map.yaml ./sparse_opt/sparse.yaml ./prob/conv1d_sparse.prob.yaml -o output_sparse_only/

tail -n 30 output_sparse_only/timeloop-model.stats.txt
