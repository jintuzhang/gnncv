#!/bin/sh
export PLUMED_MAXBACKUP=1000000
rm -rf grids
mkdir grids
reweight.py --colvar colvar --cv x --stride 100 --outfile grids/ff.dat --sigma 0.01 --temp 300
python fe.py > fe.c
rm -rf grids
