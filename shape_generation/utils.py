from datasets import HexagonDataset, HexagonRegressionDataset
from torch.utils.data import DataLoader
import torch
import numpy as np

def get_dataloaders(args):
    n_samples = int(100000 * args.dataset_frac)
    print(f"Number of samples: {n_samples}")
    train_dataset = HexagonDataset(num_samples=n_samples, scale_range=args.scale)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = HexagonDataset(num_samples=10000, scale_range=args.scale)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    return {"train": train_dataloader, "valid": valid_dataloader}

def get_dataloaders_regression(args):
    n_samples = int(100000 * args.dataset_frac)
    print(f"Number of samples: {n_samples}")
    train_dataset = HexagonRegressionDataset(num_samples=n_samples, scale_range=args.scale, num_vertices=args.num_vertices, aug_angle=args.aug_angle)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = HexagonRegressionDataset(num_samples=10000, scale_range=args.scale, num_vertices=args.num_vertices, aug_angle=args.aug_angle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    samples_dataset = HexagonRegressionDataset(num_samples=10000, scale_range=args.scale, num_vertices=args.num_vertices, aug_angle=args.aug_angle)
    samples_dataloader = DataLoader(samples_dataset, batch_size=1, shuffle=True)
    return {"train": train_dataloader, "valid": valid_dataloader, 'samples': samples_dataloader}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train a diffusion model on hexagons")
    parser.add_argument("--loss_mode", type=str, default="energy", help="Loss mode for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--wandb", type=bool, default=False, help="Use wandb for logging")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--dataset_frac", type=float, default=1.0, help="Number of diffusion timesteps")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size for training")
    parser.add_argument("--coeff_mode", type=str, default="exp_dist", help="Mode for energy coefficients")
    parser.add_argument("--best_hypers", type=bool, default=False, help="Use best hyperparameters")
    parser.add_argument('--scale', type=eval, default=[0.3, 10], help='Scale range for the dataset')
    parser.add_argument('--base_loss_mode', type=str, default="mse")
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_vertices', type=int, default=20)
    parser.add_argument('--edge_mode', type=str, default='complete')
    parser.add_argument('--aug_angle', type=float, default=-1, help='Augmentation angle for the dataset')
    return parser.parse_args()

def get_wandb_name(args):
    name = f"v{args.num_vertices}_loss_{args.loss_mode}_lr_{args.lr}_seed_{args.seed}_dataset_{args.dataset_frac}"
    return name

def load_best_hyperparameters(args):
    load_best_hyperparameters = {
        "energy": {
            "constant": {
                2.0: {"lr": 0.1},
                1.0: {"lr": 0.05},
                0.1: {"lr": 0.001},
                0.01: {"lr": 0.005},
                0.001: {"lr": 0.005},
            },
            "exp_dist": {
                2.0: {"lr": 0.1},
                1.0: {"lr": 0.1},
                0.1: {"lr": 0.01},
                0.01: {"lr": 0.01},
                0.001: {"lr": 0.05},
            }
        },
        "mse": { 
            2.0: {"lr": 0.01},
            1.0: {"lr": 0.01},
            0.1: {"lr": 0.01},
            0.01: {"lr": 0.01},
            0.001: {"lr": 0.01},
        },
        "hybrid": { 
            "constant": {
                2.0: {"lr": 0.005},
                1.0: {"lr": 0.005},
                0.1: {"lr": 0.005},
                0.01: {"lr": 0.005},
                0.001: {"lr": 0.005},
            },
            "exp_dist": {
                2.0: {"lr": 0.005},
                1.0: {"lr": 0.005},
                0.1: {"lr": 0.01},
                0.01: {"lr": 0.005},
                0.001: {"lr": 0.005},
            }
        },
        "fape": { 
            2.0: {"lr": 0.1},
            1.0: {"lr": 0.1},
            0.1: {"lr": 0.05},
            0.01: {"lr": 0.005},
            0.001: {"lr": 0.005},
        }
    }

    if args.loss_mode in ["energy", "hybrid"]:
        best_hyperparameters = load_best_hyperparameters[args.loss_mode][args.coeff_mode][args.dataset_frac]
    else:
        best_hyperparameters = load_best_hyperparameters[args.loss_mode][args.dataset_frac]

    args.lr = best_hyperparameters["lr"]
    print(f"Loaded best hyperparameters.")

    return args

def get_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm