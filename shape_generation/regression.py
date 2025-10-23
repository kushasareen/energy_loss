import wandb
from utils import set_seed, parse_args, get_wandb_name, load_best_hyperparameters, get_dataloaders_regression
from models_regression import PredictionModel

def main():
    args = parse_args()
    if args.best_hypers:
        args = load_best_hyperparameters(args)

    set_seed(args.seed)
    if args.wandb:
        wandb.init(project="hexagon_regression", name= get_wandb_name(args), dir = "/home/mila/k/kusha.sareen/scratch/shape_generation/wandb")
        wandb.config.update(args)
    dataloaders = get_dataloaders_regression(args)
    predictor = PredictionModel(args)
    predictor.train(dataloaders, epochs=args.epochs)
    predictor.visualize_samples(dataloaders, 10)

if __name__ == "__main__":
    main()
