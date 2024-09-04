#!/bin/sh

plumed driver --mf_dcd ../run_unbiased/r/r.dcd --plumed ./plumed.inp
mv colvar colvar.A
plumed driver --mf_dcd ../train_gnn/B.dcd --plumed ./plumed.inp
mv colvar colvar.B
plumed driver --mf_dcd ../run_unbiased/p/p.dcd --plumed ./plumed.inp
mv colvar colvar.C
