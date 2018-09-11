#PBS -qgpu
#PBS -lwalltime=00:30:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250G
#PBS -S /bin/bash

cd /home/lgpu0009/code
rm ./*.sh.*

source activate base
export PYTHONPATH=home/lgpu0009/code

python train_convnet_pytorch.py
