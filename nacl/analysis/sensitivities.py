import torch
import numpy as np
import mdtraj as md
import mlcolvar.graph as mg

mg.utils.torch_tools.set_default_dtype('float32')

dataset = mg.utils.io.create_dataset_from_trajectories(
    trajectories=['../run_unbiased_long/traj.dcd'],
    top=['../data/solvate.psf'],
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

cv.load_state_dict(torch.load('../train/6A_2layer_1c/model.pt'))
cv = cv.eval()

s = mg.explain.graph_node_sensitivity(
    cv,
    dataset,
    device='cuda',
    batch_size=1000
)

np.savetxt('sensitivities.dat', s['sensitivities'])

np.savetxt(
    'max-sensitivities.dat',
    np.max(s['sensitivities_components'][:, 2:], axis=1)
)

traj = md.load('../data/r.pdb', top='../data/solvate.psf')
traj = traj.atom_slice(traj.top.select('not type H'))

index_o = traj.top.select('type O')
index_na = traj.top.select('type Na')
pairs_o_na = np.zeros((len(index_o), 2), dtype=int)
pairs_o_na[:, 0] = index_na
pairs_o_na[:, 1] = index_o

distances = []
sensitivities_components = []

for i, d in enumerate(dataset):
    traj.xyz[0, :, :] = d.to_dict()['positions'].detach().cpu().numpy() / 10

    distances_o_na = md.compute_distances(traj, pairs_o_na).T

    order = np.argsort(distances_o_na.T)

    distances.append(distances_o_na[order][0])
    sensitivities_components.append(
        s['sensitivities_components'][i, 2:][order]
    )

distances = np.mean(np.hstack(distances), axis=1)
sensitivities = np.mean(
    np.array(sensitivities_components), axis=0
)

np.savetxt(
    'distance-sensitivities_o.dat',
    np.hstack([np.array([distances]).T, sensitivities.T])
)
