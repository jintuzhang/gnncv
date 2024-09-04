import torch
import mlcolvar.graph as mg
from mlcolvar.utils.plot import plot_metrics

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlcolvar.utils.trainer import MetricsCallback

from matplotlib import pyplot as plt

mg.utils.torch_tools.set_default_dtype('float32')

dataset = mg.utils.io.create_dataset_from_trajectories(
    trajectories=['../../run_unbiased/long/A.dcd'],
    top=['../../data/A.gro'],
    cutoff=10,  # Ang
    create_labels=True,
    system_selection='not type H',
)
mg.data.save_dataset(dataset, 'dataset.pt')
datasets = mg.utils.timelagged.create_timelagged_datasets(
    dataset, lag_time=2
)
datamodule = mg.data.GraphCombinedDataModule(
    datasets, random_split=False, batch_size=5000
)
print(datamodule)

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

metrics = MetricsCallback()
early_stopping = EarlyStopping(
    monitor='train_loss', patience=20, min_delta=1e-3
)
trainer = Trainer(
    callbacks=[metrics, early_stopping],
    logger=None,
    enable_checkpointing=False,
    accelerator='gpu',
    max_epochs=500
)

trainer.fit(cv, datamodule)
cv = cv.eval()

torch.save(cv.state_dict(), 'model.pt')
cv.to_torchscript('model.ptc')

plot_metrics(
    metrics.metrics,
    keys=[x for x in metrics.metrics.keys()],
    yscale='log'
)

plt.show()

datamodule = mg.data.GraphDataModule(dataset, lengths=(1,))
datamodule.setup()

fig, ax = plt.subplots(figsize=(5, 5))
with torch.no_grad():
    data = next(iter(datamodule.train_dataloader()))
    s = cv(data)
    ax.hist(s.numpy(), bins=100)

ax.set_xlabel('CV')
ax.set_ylabel('Histogram')
ax.legend()
plt.show()
