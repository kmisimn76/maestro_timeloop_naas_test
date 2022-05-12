echo "Usage: ./run_vgg.sh [layer number]"
../../build/timeloop-model ./arch/1level.arch.yaml ./prob/vgg${1}.prob.yaml ./map/vgg${1}.map.yaml
cat timeloop-model.stats.txt | grep Cycles: >> result.txt
