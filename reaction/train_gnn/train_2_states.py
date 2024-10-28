import torch
import mlcolvar.graph as mg
from mlcolvar.utils.plot import plot_metrics

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlcolvar.utils.trainer import MetricsCallback

from matplotlib import pyplot as plt

mg.utils.torch_tools.set_default_dtype('float64')

dataset = mg.utils.io.create_dataset_from_trajectories(
    trajectories=['../run_unbiased/r/r.dcd', '../run_unbiased/p/p.dcd'],
    top=['../data/r.pdb', '../data/p.pdb'],
    cutoff=10.0,  # Ang
    create_labels=True,
    system_selection='all and not type H',
)
datamodule = mg.data.GraphDataModule(dataset, shuffle=[1, 0])
print(datamodule)

cv = mg.cvs.GraphDeepTDA(
    n_cvs=1,
    cutoff=dataset.cutoff,
    atomic_numbers=dataset.atomic_numbers,
    target_centers=[-7, 7],
    target_sigmas=[1, 1],
    model_options={
        'n_bases': 8,
        'n_polynomials': 6,
        'n_layers': 1,
        'n_messages': 2,
        'n_feedforwards': 2,
        'n_scalars_node': 8,
        'n_vectors_node': 8,
        'n_scalars_edge': 8,
        'drop_rate': 0.1,
        'activation': 'SiLU',
    },
)

metrics = MetricsCallback()
early_stopping = EarlyStopping(
    monitor='valid_loss', patience=250, min_delta=1e-5
)
trainer = Trainer(
    callbacks=[metrics, early_stopping],
    logger=None,
    enable_checkpointing=False,
    accelerator='gpu',
    max_epochs=3000
)

trainer.fit(cv, datamodule)
cv = cv.eval()

torch.save(cv.state_dict(), 'model.pt')
cv.to_torchscript('model.ptc')

plot_metrics(
    metrics.metrics,
    keys=['train_loss_epoch', 'valid_loss'],
    linestyles=['-.', '-'], colors=['fessa1', 'fessa5'],
    yscale='log'
)

plt.show()

fig, ax = plt.subplots(figsize=(5, 5))
with torch.no_grad():
    ax.hist(
        cv(next(iter(datamodule.train_dataloader()))).numpy(),
        bins=100,
        alpha=0.5,
        label='train'
    )
    ax.hist(
        cv(next(iter(datamodule.val_dataloader()))).numpy(),
        bins=100,
        alpha=0.5,
        label='val'
    )

ax.set_xlabel('CV')
ax.set_ylabel('Histogram')
ax.legend()
plt.show()
