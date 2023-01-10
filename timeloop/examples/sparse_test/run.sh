../../build/timeloop-model ./arch/1level.arch.yaml ./prob/conv1d.prob.yaml ./map/conv1d-1level.map.yaml -o output/
tail -n 30 output/timeloop-model.stats.txt
