#!/bin/sh
export PLUMED_MAXBACKUP=1000000
rm -rf grids
mkdir grids
reweight.py --colvar colvar --cv d --stride 100 --outfile grids/ff.dat --sigma 0.01 --temp 300 --min 0.1 --max 0.6
python fe.py > fe
rm -rf grids
