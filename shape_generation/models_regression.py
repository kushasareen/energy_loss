import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from kabsch import kabsch_torch_batched
from losses import compute_loss
from utils import get_gradient_norm
from torch.autograd import Variable

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple MLP denoiser model
class MLPPredictor(nn.Module):
    def __init__(self, input_dim=1, output_dim = 40, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class PredictionModel:
    def __init__(self, args, criterion=nn.MSELoss()):
        denoiser = MLPPredictor(output_dim=2 * args.num_vertices)
        self.model = denoiser.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = criterion
        self.test_epochs = 10
        self.loss_mode = args.loss_mode
        self.coeff_mode = args.coeff_mode
        self.args = args

    def train(self, dataloaders, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_grad_norm = 0
            for batch in dataloaders["train"]:
                batch = [b.to(device) for b in batch]
                params, data = batch
                self.optimizer.zero_grad()
                pred = self.model(params)

                loss = compute_loss(pred, data, loss_mode=self.loss_mode, coeff_mode=self.coeff_mode, base_loss_mode=self.args.base_loss_mode,  num_vertices=self.args.num_vertices, edge_mode=self.args.edge_mode)
                loss.backward()
                grad_norm = get_gradient_norm(self.model)
                self.optimizer.step()

                total_loss += loss.item()
                total_grad_norm += grad_norm
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloaders['train']):.6f}, Grad Norm: {total_grad_norm / len(dataloaders['train']):.4f}")

            if (epoch + 1) % self.test_epochs == 0:
                self.validate(dataloaders)

    def validate(self, dataloaders):
        self.model.eval()
        total_loss = 0
        total_mse = 0
        with torch.no_grad():
            for batch in dataloaders["valid"]:
                batch = [b.to(device) for b in batch]
                params, data = batch
                self.optimizer.zero_grad()
                pred = self.model(params)

                loss = compute_loss(pred, data, loss_mode=self.loss_mode, coeff_mode=self.coeff_mode, base_loss_mode=self.args.base_loss_mode, num_vertices=self.args.num_vertices, edge_mode=self.args.edge_mode)
                mse = self.criterion(pred, data)
                total_mse += mse.item()
                total_loss += loss.item()

        print(f"Valid Loss: {total_loss / len(dataloaders['valid'])}")
        print(f"Valid MSE: {total_mse / len(dataloaders['valid'])}")
        diversity_metric, quality_metric, overall_metric, samples_grad = self.get_metrics(dataloaders)
        print(f"Diversity Metric: {diversity_metric:.6f}, Quality Metric: {quality_metric:.6f}, Overall Metric: {overall_metric:.6f}, Samples Grad Norm: {samples_grad:.6f}")
        if wandb.run:
            wandb.log({"Valid Loss": total_loss / len(dataloaders["valid"]),
                        "Valid MSE": total_mse / len(dataloaders["valid"]),
                        "Diversity Metric": diversity_metric,
                        "Quality Metric": quality_metric,
                        "Overall Metric": overall_metric,
                        "Samples Grad Norm": samples_grad})

    def sample(self, dataloaders, num_samples=10): # probably fine
        self.model.eval()
        samples = []
        sample_idx = 0
        with torch.no_grad():
            for batch in dataloaders["samples"]:
                params, _ = batch
                pred = self.model(params)
                samples.append(pred)
                sample_idx += len(pred)

                if sample_idx >= num_samples:
                    break

        samples = torch.stack(samples).view(-1, self.args.num_vertices, 2)
        return samples.view(num_samples, self.args.num_vertices, 2).cpu()

    def visualize_samples(self, dataloaders, num_samples=5, filename="hexagons"):
        samples = self.sample(dataloaders, num_samples).detach().numpy()
        plt.figure(figsize=(6, 6), dpi=150)
        for i, sample in enumerate(samples):
            plt.scatter(sample[:, 0], sample[:, 1], label=f"Sample {i+1}")
        plt.title("Generated Hexagonal Point Clouds")
        plt.savefig(f"images/{filename}.png")

    def get_metrics(self, dataloaders, num_samples=1000): # lower is better
        samples = self.sample(dataloaders, num_samples)
        samples_grad = self.log_gradient_wrt_samples(dataloaders)
        samples = samples.detach().numpy()
        center_of_mass = np.mean(samples, axis=1)
        centered_samples = samples - center_of_mass[:, None]
        sample_norms = np.linalg.norm(centered_samples, axis=2)
        total_scores = 0

        for centered_sample in centered_samples:
            angles = np.arctan2(centered_sample[:, 1], centered_sample[:, 0])
            angles = np.sort(angles)
            angle_differences = np.diff(np.concatenate((angles, [angles[0] + 2 * np.pi])))
            angle_std = np.std(angle_differences)
            radial_std = np.std(np.linalg.norm(centered_sample, axis=1))
            sample_scale = np.linalg.norm(centered_sample, axis=1).mean()
            sample_score = angle_std / (2 * np.pi) + radial_std / sample_scale
            total_scores += sample_score
        
        diversity_metric = -np.log(sample_norms.mean(-1).std()) # lower is better
        quality_metric = np.log(total_scores / num_samples)
        overall_metric = diversity_metric + quality_metric
        return diversity_metric, quality_metric, overall_metric, samples_grad

    def log_gradient_wrt_samples(self, dataloaders, num_samples=1000):
        self.model.train()
        total_grad = 0
        sample_idx = 0
        for batch in dataloaders["samples"]:
            batch = [b.to(device) for b in batch]
            params, data = batch
            self.optimizer.zero_grad()
            pred = self.model(params)

            loss = compute_loss(pred, data, loss_mode=self.loss_mode, coeff_mode=self.coeff_mode, base_loss_mode=self.args.base_loss_mode,  num_vertices=self.args.num_vertices, edge_mode=self.args.edge_mode)

            pred.retain_grad()
            loss.backward()
            grad = pred.grad
            total_grad += grad.norm().item()

            sample_idx += len(pred)
            if sample_idx >= num_samples:
                break
            
        return total_grad / sample_idx



        

        


if __name__ == "__main__":
    model = MLPPredictor()
    print(model)
    print(model(torch.randn(32, 1)))