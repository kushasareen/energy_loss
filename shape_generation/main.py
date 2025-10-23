import wandb
from utils import set_seed, parse_args, get_wandb_name, load_best_hyperparameters, get_dataloaders
from models_noise_schedule import DiffusionModel, MLPDenoiser

def main():
    args = parse_args()
    if args.best_hypers:
        args = load_best_hyperparameters(args)

    set_seed(args.seed)
    if args.wandb:
        wandb.init(project="hexagons", name= get_wandb_name(args), dir = "/home/mila/k/kusha.sareen/scratch/shape_generation/wandb")
        wandb.config.update(args)
    dataloaders = get_dataloaders(args)
    denoiser = MLPDenoiser(hidden_dim=args.hidden_dim)
    diffusion = DiffusionModel(args, denoiser=denoiser)
    diffusion.train(dataloaders, epochs=args.epochs)
    diffusion.visualize_samples(10)

if __name__ == "__main__":
    main()
