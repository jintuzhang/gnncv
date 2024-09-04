#!/bin/sh

plumed driver --mf_dcd ../train_gnn/A.dcd --plumed ./plumed.inp
mv colvar colvar.A
plumed driver --mf_dcd ../train_gnn/B.dcd --plumed ./plumed.inp
mv colvar colvar.B
plumed driver --mf_dcd ../train_gnn/C.dcd --plumed ./plumed.inp
mv colvar colvar.C
