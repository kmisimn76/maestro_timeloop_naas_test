#LEN=12
##LEN=52

#for i in {0..${LEN}};do
#    echo ${i}
#    ../../timeloop/build/timeloop-model ./best/${i}/hw_.yaml ./best/${i}/mapping_.yaml ./best/${i}/sparse_.yaml ./best/${i}/problem_.yaml -o output_tmp/
#    grep "Cycles: " output_tmp/timeloop-model.stats.txt
#done

#echo "result"
#for i in {0..${LEN}};do
#    ../../timeloop/build/timeloop-model ./best/${i}/hw_.yaml ./best/${i}/mapping_.yaml ./best/${i}/sparse_.yaml ./best/${i}/problem_.yaml -o output_tmp/ > /dev/null
#    grep "Cycles: " output_tmp/timeloop-model.stats.txt
#done

for entry in `ls -d best/*`; do
    echo $entry
    cd $entry
    pwd
    ../../../../timeloop/build/timeloop-model ./hw_.yaml ./mapping_.yaml ./sparse_.yaml ./problem_.yaml
    grep "Cycles: " timeloop-model.stats.txt
    cd ../..
    echo ""
done
echo "result"
for entry in `ls -d best/*`; do
    cd $entry
    grep "Cycles: " timeloop-model.stats.txt
    cd ../..
done
echo "L1 size"
cat best/*/timeloop-model.map.txt  | grep L1
echo "L2 size"
cat best/*/timeloop-model.map.txt  | grep L2

