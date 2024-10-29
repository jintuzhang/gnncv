import torch
import numpy as np

import mlcolvar.graph as mg
from mlcolvar.explain.lasso import lasso_regression
from mlcolvar.utils.io import create_dataset_from_files, DictDataset

mg.utils.torch_tools.set_default_dtype('float32')

dataset = mg.utils.io.create_dataset_from_trajectories(
    trajectories=['../../run_unbiased_long/traj.dcd'],
    top=['../../data/solvate.psf'],
    cutoff=6,  # Ang
    create_labels=True,
    system_selection='all and not type H',
)

cv = mg.cvs.GraphDeepTICA(
    n_cvs=1,
    cutoff=dataset.cutoff,
    atomic_numbers=dataset.atomic_numbers,
    model_options={
        'n_out': 8,
        'n_bases': 8,
        'n_polynomials': 6,
        'n_layers': 2,
        'n_messages': 2,
        'n_feedforwards': 2,
        'n_scalars_node': 6,
        'n_vectors_node': 4,
        'n_scalars_edge': 6,
        'drop_rate': 0.0,
        'activation': 'Tanh',
    },
    optimizer_options={'optimizer': {'lr': 4E-3, 'weight_decay': 1E-4}}
)

cv.load_state_dict(torch.load('../../train/6A_2layer_1c/model.pt'))
cv = cv.eval()

target = -mg.explain.utils.get_dataset_cv_values(cv, dataset).squeeze()
target = (target - target.min()) / (target.max() - target.min()) * 2 - 1

_, df = create_dataset_from_files(
    './colvar', return_dataframe=True
)
mask = df['d'] <= 0.7

n_b = np.loadtxt('n_b.dat')
df['n_b'] = n_b

for k in df.keys():
    df[k] = (df[k] - df[k].min()) / (df[k].max() - df[k].min()) * 2 - 1
df = df[mask]
target = target[mask]
df = df.filter(regex='d|c_o|c_h|n_b')
dataset_ff = DictDataset({'data': df.values})
dataset_ff.feature_names = df.columns.values

with torch.no_grad():
    dataset_ff['target'] = torch.tensor(target)

regressor, feats, coeffs = lasso_regression(dataset_ff, plot=True)
