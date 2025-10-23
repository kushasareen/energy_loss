# a script to plot wandb exports for the paper
# will also contain as input a hand made file with eval logs

import numpy as np
import pandas as pd

# Figures to make:
# 1.  Molecule stability vs epoch
# rigidity stuff? pareto curve?


def get_epoch_from_iteration(iteration, model):
    if model == 'gnn_dynamics':
        batch_size = 512
    elif model == 'gnn_dynamics':
        batch_size = 400
    else:
        raise ValueError(f'Unknown model {model}')

    return np.ceil(iteration / batch_size)


def main():
    wandb_exports = []
    pass

if __name__ == '__main__':
    main()