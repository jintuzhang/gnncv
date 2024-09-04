#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=2:mpiprocs=2:ngpus=1
#PBS -q gpu_a100
#PBS -N alad_run
#PBS -o alad_run.out
#PBS -e alad_run.err

cd $PBS_O_WORKDIR

export BLAS=MKL
export USE_MKLDNN=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PLUMED_NUM_THREADS=1
export OPENMM_NUM_THREADS=1

source /work/jzhang/softwares/plumed.python/bin/activate

python ./omm.py > log
