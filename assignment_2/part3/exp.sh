#!/bin/bash

#PBS -qgpu
#PBS -lwalltime=04:00:00
#PBS -S /bin/bash
#PBS -lnodes=1:ppn=12
#PBS -lmem=250G

cd /home/lgpu0009/assignment_2/part3
rm ./*.sh.*

source activate base
export PYTHONPATH=home/lgpu0009/assignment_2/part3

python train.py