#!/bin/bash

export BLAS=MKL
export USE_MKLDNN=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PLUMED_NUM_THREADS=1
export OPENMM_NUM_THREADS=6

mdconvert ../run_biased/biased.trajectory.h5 -fo traj.dcd
plumed driver --mf_dcd traj.dcd --plumed plumed.inp
