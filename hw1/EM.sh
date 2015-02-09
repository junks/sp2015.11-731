#!/bin/sh
N=10000

rm -f results/EM_*
rm -f outputs/output_*

rm -f AER.txt
touch AER.txt

for i in `seq 1 5`;
do
    echo "time ./align -n $N -i $i > output.txt && cat output.txt | ./check | ./grade > results/EM_$i.txt" 
    time ./align -n $N -i $i > output.txt && cat output.txt | ./check | ./grade > results/EM_$i.txt
    
    echo "cp output.txt outputs/output_$i.txt"
    cp output.txt outputs/output_$i.txt

    tail --lines 3 results/EM_$i.txt

    tail --lines 1 results/EM_$i.txt >> AER.txt

    echo
done


tail --lines 3 results/*


