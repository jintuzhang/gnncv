import plumed
import numpy as np
for i in range(1, 2000):
    f1 = plumed.read_as_pandas('./grids/ff_{}.dat'.format(i))
    f11 = f1[f1['d'] < 0.35]
    f12 = f1[f1['d'] > 0.35]
    f12 = f12[f12['d'] < 0.6]
    fesA = -2.49 * np.logaddexp.reduce(-1 / 2.49 * f12['file.free'])
    fesB = -2.49 * np.logaddexp.reduce(-1 / 2.49 * f11['file.free'])
    print(i * 0.02, fesA - fesB)
