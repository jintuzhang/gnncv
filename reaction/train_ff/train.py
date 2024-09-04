import torch
from mlcolvar.data import DictModule
from mlcolvar.utils.io import create_dataset_from_files
from mlcolvar.cvs import DeepTDA

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlcolvar.utils.trainer import MetricsCallback

from matplotlib import pyplot as plt
from mlcolvar.utils.plot import plot_metrics


filenames = ['colvar.A', 'colvar.B', 'colvar.C']

dataset, df = create_dataset_from_files(
    filenames, return_dataframe=True, filter_args={'regex': 'd*'}
)

datamodule = DictModule(dataset, lengths=[0.8, 0.2])
print(datamodule)

cv = DeepTDA(
    n_states=len(filenames),
    n_cvs=1,
    target_centers=[-7, 1, 7],
    target_sigmas=[0.5, 1, 0.5],
    layers=[21, 48, 12, 1]
)

metrics = MetricsCallback()
early_stopping = EarlyStopping(
    monitor='train_loss', patience=250, min_delta=1e-5
)
trainer = Trainer(
    callbacks=[metrics, early_stopping],
    logger=None,
    enable_checkpointing=False,
    max_epochs=3000
)

trainer.fit(cv, datamodule)
cv = cv.eval()

torch.save(cv.state_dict(), 'model.pt')
cv.to_torchscript('model.ptc', method='trace')

plot_metrics(
    metrics.metrics,
    keys=['train_loss_epoch', 'valid_loss'],
    linestyles=['-.', '-'], colors=['fessa1', 'fessa5'],
    yscale='log'
)

plt.show()

fig, ax = plt.subplots(figsize=(5, 5))
X = dataset[:]['data']
Y = dataset[:]['labels']

with torch.no_grad():
    s = cv(torch.Tensor(X)).numpy()

for i in range(0, len(filenames)):
    s_red = s[torch.nonzero(Y == i, as_tuple=True)]
    ax.hist(s_red[:, 0], bins=100, label=f'State {i}')

ax.set_xlabel('CV')
ax.set_ylabel('Histogram')
ax.legend()
plt.show()
