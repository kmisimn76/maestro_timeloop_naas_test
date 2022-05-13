#../../build/timeloop-model ./arch/1level_sparse.arch.yaml ./comp/metadata.yaml ./map/conv1d-1level-sparse.map.yaml ./sparse_opt/sparse.yaml ./prob/conv1d_sparse.prob.yaml -o output_sparse_only/
../../build/timeloop-model ./arch/sparse_npu.arch.yaml ./comp/metadata.yaml ./map/sparse_npu.map.yaml ./sparse_opt/sparse.yaml ./prob/conv1d_sparse.prob.yaml -o output_sparse/

tail -n 30 output_sparse_only/timeloop-model.stats.txt
