for i in {0..12};do
    echo ${i}
    ../../timeloop/build/timeloop-model ./best/${i}/hw_.yaml ./best/${i}/mapping_.yaml ./best/${i}/sparse_.yaml ./best/${i}/problem_.yaml -o output_tmp/
    grep "Cycles: " output_tmp/timeloop-model.stats.txt
done
