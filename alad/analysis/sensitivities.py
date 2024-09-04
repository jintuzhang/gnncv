import torch
import numpy as np
import mlcolvar.graph as mg

mg.utils.torch_tools.set_default_dtype('float32')

dataset = mg.data.load_dataset('../train/10A_2layer_1c/dataset.pt')

cv = mg.cvs.GraphDeepTICA(
    n_cvs=1,
    cutoff=dataset.cutoff,
    atomic_numbers=dataset.atomic_numbers,
    model_options={
        'n_out': 6,
        'n_bases': 10,
        'n_polynomials': 6,
        'n_layers': 2,
        'n_messages': 2,
        'n_feedforwards': 2,
        'n_scalars_node': 6,
        'n_vectors_node': 2,
        'n_scalars_edge': 6,
        'drop_rate': 0.0,
        'activation': 'Tanh',
    },
    optimizer_options={
        'optimizer': {'lr': 3E-3, 'weight_decay': 2E-4},
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.ExponentialLR,
            'gamma': 0.9995
        }
    }
)

cv.load_state_dict(torch.load('../train/10A_2layer_1c/model.pt'))
cv = cv.eval()

s = mg.explain.graph_node_sensitivity(cv, dataset, device='cuda')

np.savetxt('sensitivities.dat', s['sensitivities'])
