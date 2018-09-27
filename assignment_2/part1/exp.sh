#!/bin/bash

#PBS -qgpu
#PBS -lwalltime=10:00:00
#PBS -S /bin/bash
#PBS -lnodes=1:ppn=12
#PBS -lmem=250G


cd /home/lgpu0009/assignment_2/part1
rm ./*.sh.*

source activate base
export PYTHONPATH=home/lgpu0009/assignment_2/part1

for k in 128
do
        for i in "RNN" "LSTM" 
        do
                for j in 50 40 30 20 10 5 6 7 8 9 11 12 13 14 15
                do
                python train.py --input_length $j --model_type $i --batch_size $k
                done
        done
done