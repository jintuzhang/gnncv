#!/bin/sh
export PLUMED_MAXBACKUP=1000000
rm -rf grids
mkdir grids
reweight.py --colvar colvar --cv phi --stride 500 --outfile grids/ffphi.dat --sigma 0.05 --temp 300
python fe.py > fe.phipsi
rm -rf grids
