#!/bin/bash

mdconvert ../run_biased/biased.trajectory.h5 -fo traj.dcd
plumed driver --mf_dcd traj.dcd --plumed plumed.inp
python split.py
