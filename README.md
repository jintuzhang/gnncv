Training and simulation inputs for paper "Descriptors-free Collective Variables From Geometric Graph Neural Networks".

Link to the paper: https://pubs.acs.org/doi/10.1021/acs.jctc.4c01197 (arXiv: https://arxiv.org/abs/2409.07339v3)

---

The contents are organized as follows:
- **plumed_pytorch_gnn**: contains the plumed interface for GNN-based CVs
- **alad**: contains the files to reproduce alanine dipeptide in a vacuum results
    - **data**: topology and force field files
    - **analysis**: analysis scripts
    - **train**: scripts for the training of the GNN-CVs and trained model
    - **run_biased**: simulation files for biased simulations using phi and psi as CVs
    - **run_biased_gnn/10A_2layer_1c/1**: simulation files for biased simulations using GNN-CV
    - **run_unbiased/long**: simulation files for unbiased simulations
- **nacl**: contains the files reproduce NaCl dissociation in explicit water results
    - **data**: topology and force field files
    - **analysis**: analysis scripts
    - **train**: scripts for the training of the GNN-CVs and trained model
    - **run_biased**: simulation files for biased simulations using interionic distance and oxygen coordination of Na+ as CVs
    - **run_biased_gnn/6A_2layer_1c/1**: simulation files for biased simulations using GNN-CV
    - **run_unbiased/long**: simulation files for unbiased simulations
- **reaction**: contains the files reproduce methyl migration of FDMB cation results
    - **data**: topology files
    - **eval**: plumed files to evaluate GNN-CV using plumed driver
    - **run_biased**: simulation files for biased simulations using coordiantion difference as CV
    - **run_biased_gnn**: simulation files for biased simulations using GNN-CV
    - **run_unbiased**: simulation files for unbiased simulations
    - **train_ff**: scripts for the training of MLCV based on feed-forward NN and trained model
    - **train_ff_full**: scripts for the training of MLCV based on feed-forward NN and a fully permutated dataset and trained model
    - **train_gnn**: scripts for the training of the GNN-CVs and trained model

---

The modified version of the `mlcolvar` library used for the GNN-CV training is available at: https://github.com/jintuzhang/mlcolvar

The relevant code for the GNN-CV definiton and training is implemented in the `mlcolvar.graph` module of such a library.
