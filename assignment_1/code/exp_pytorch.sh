#PBS -qgpu
#PBS -lwalltime=00:01:00

source activate base
python train_ml_pytorch.py