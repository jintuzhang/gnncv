import torch
import numpy as np
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

np.savetxt(
    'max-sensitivities_nacl.dat',
    np.max(s['sensitivities_components'][:, :2], axis=1)
)
