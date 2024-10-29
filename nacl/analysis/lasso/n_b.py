import numpy as np
import mdtraj as md

from mlcolvar.graph.utils.progress import pbar

f_na_o = lambda d: 0.5 * (1 - np.tanh(3 * (d * 10 - 3.25)))
f_cl_h = lambda d: 0.5 * (1 - np.tanh(3 * (d * 10 - 3.10)))

t = md.load_dcd('../../run_unbiased_long/traj.dcd', top='../../data/r.pdb')

results = []
for f in pbar(t, prefix='n_b', frequency=0.0001):
    tmp = 0
    for r in list(t.top.residues)[2:]:
        d_na_o = md.compute_distances(
            f, [[0, r.atom(0).index]]
        )
        d_cl_h = md.compute_distances(
            f, [[1, r.atom(1).index], [1, r.atom(2).index]]
        )

        d_na_o = f_na_o(d_na_o[0, 0])
        d_cl_h = f_cl_h(min([d_cl_h[0, 0], d_cl_h[0, 1]]))

        tmp += min(d_na_o, d_cl_h)

    results.append(tmp)

np.savetxt('n_b.dat', results)
