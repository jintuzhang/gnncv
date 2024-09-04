import torch
import numpy as np

from cycler import cycler
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import mlcolvar.graph as mg
from mlcolvar.data.dataset import DictDataset
from mlcolvar.explain.lasso import lasso_regression
from mlcolvar.utils.io import create_dataset_from_files

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

target = -mg.explain.utils.get_dataset_cv_values(cv, dataset).squeeze()
target = (target - target.min()) / (target.max() - target.min()) * 2 - 1

_, df = create_dataset_from_files(
    '../run_unbiased/long/descriptors', return_dataframe=True
)

shifts = {'phi': 2, 'psi': -2.2, 'omega': 0}
for k, v in shifts.items():
    df[k] = df[k].transform(lambda x: x - 2 * np.pi if x > v else x)
for k in ['phi', 'psi', 'theta', 'omega']:
    df[k] = (df[k] - df[k].min()) / (df[k].max() - df[k].min()) * 2 - 1
df = df.filter(regex='^phi|^psi|^theta|^omega')
# df = df.filter(regex='^sin')
dataset_ff = DictDataset({'data': df.values})
dataset_ff.feature_names = df.columns.values
with torch.no_grad():
    dataset_ff['target'] = torch.tensor(target)

regressor, feats, coeffs = lasso_regression(dataset_ff, plot=True)

num_features = [1, 2, 3]
num_features_path_ = np.count_nonzero(
    np.abs(regressor.coefs_paths_.T) > 1E-4, axis=1
)

print(num_features_path_)

selected_alphas = []
for num_feat in num_features:
    id = np.argwhere(num_features_path_ == num_feat).max() - 1
    alpha = regressor.alphas_[id]
    selected_alphas.append(alpha)
    print(f'num_feat={num_feat} --> alpha={alpha:.2e}')

with torch.no_grad():
    X = dataset_ff['data'].numpy()
    y = dataset_ff['target'].numpy()

X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

# test different values of alpha
fig, ax = plt.subplots()
ax.set_prop_cycle(cycler(color=plt.get_cmap('Dark2').colors))

for alpha in selected_alphas:
    regressor, feats, coeffs = lasso_regression(
        dataset_ff,
        alphas=[alpha],
        print_info=True,
        plot=False,
    )
    y_pred = regressor.predict(X)
    ax.scatter(
        y_pred[::5],
        y[::5],
        s=5,
        label=f'n_features={len(coeffs)} (alpha={alpha:.2e})',
        alpha=0.5
    )
    # print equation
    equation = "y="
    for f, c in zip(feats, coeffs):
        equation += f"+{c:.3f}*{f} " if c > 0 else f"{c:.3f}*{f} "

    print(equation[:-1])
    print('\n')

ax.set_xlabel('Linear model')
ax.set_ylabel('MLCV')
ax.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linewidth=2,
    color='lightgrey',
    zorder=0,
    linestyle='dashed'
)
ax.legend(frameon=False)
# plt.show()
