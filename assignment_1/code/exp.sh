#!/bin/bash

#PBS -qgpu
#PBS -lwalltime=03:00:00
#PBS -S /bin/bash
#PBS -lnodes=1:ppn=12
#PBS -lmem=250G

cd /home/lgpu0009/code
rm ./*.sh.*

source activate base
export PYTHONPATH=home/lgpu0009/code

for i in "1e-7" "1e-6" "1e-5" "1e-4" 
do
	for j in 200 100 500
    do
    	python train_mlp_pytorch.py --batch_size $j --learning_rate $i
	done
done
