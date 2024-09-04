import mdtraj
import numpy as np

traj = mdtraj.load('traj.dcd', top='../data/r.pdb')
x = np.loadtxt('./colvar')[:, 1]

l_1 = []
l_2 = []
l_3 = []

for i in range(len(traj)):
    if x[i] < 0.15:
        l_1.append(i)
    elif x[i] > 0.4 and x[i] < 0.6:
        l_2.append(i)
    elif x[i] > 0.85:
        l_3.append(i)

traj[l_1].save_dcd('A.dcd')
traj[l_2].save_dcd('B.dcd')
traj[l_3].save_dcd('C.dcd')
