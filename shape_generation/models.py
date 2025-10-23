import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from kabsch import kabsch_torch_batched

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple MLP denoiser model
class MLPDenoiser(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.model(x)

# Diffusion training
class DiffusionModel:
    def __init__(self, args, denoiser = MLPDenoiser(), criterion = nn.MSELoss(), timesteps=100, noise_std=0.1):
        self.timesteps = timesteps
        self.noise_std = noise_std
        self.model = denoiser.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = criterion
        self.test_epochs = 10
        self.loss_mode = args.loss_mode # "energy" or "mse"
        self.coeff_mode = args.coeff_mode

    def compute_loss(self, denoised, data):
        mse = self.criterion(denoised, data) # Shape: (1)

        if self.loss_mode == "mse":
            loss = mse
        elif self.loss_mode == "energy":
            loss = self.compute_energy_loss(denoised.view(-1, 20, 2), data.view(-1, 20, 2)).mean() # avg over batch
        elif self.loss_mode == "hybrid":
            energy = self.compute_energy_loss(denoised.view(-1, 20, 2), data.view(-1, 20, 2)).detach() # Shape: (B) # no grads through energy
            mse_loss = (denoised - data).pow(2).mean((1)) # Shape: (B)
            loss = energy*mse_loss # Shape: (B)
            loss = loss.mean() # Shape: (1)
        elif self.loss_mode == "fape":
            loss = self.compute_fape_loss(denoised.view(-1, 20, 2), data.view(-1, 20, 2)) # Shape: (1) # TODO: Check
        return loss
    
    def compute_fape_loss(self, denoised, ground_truth):
        ground_truth_rotated, _, _ = kabsch_torch_batched(ground_truth.detach(), denoised.detach())
        fape_loss = self.criterion(ground_truth_rotated, denoised) # Shape: (1)

        return fape_loss

    def train(self, dataloaders, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for data in dataloaders["train"]:
                data = data.view(data.size(0), -1).to(device)  # Flatten (batch_size, 12)
                noise = torch.randn_like(data) * self.noise_std
                noisy_data = data + noise
                self.optimizer.zero_grad()
                denoised = self.model(noisy_data)
                loss = self.compute_loss(denoised, data)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloaders['train']):.6f}")

            if (epoch + 1) % self.test_epochs == 0:
                total_loss = 0
                total_mse = 0
                for data in dataloaders["valid"]:
                    data = data.view(data.size(0), -1).to(device)  # Flatten (batch_size, 12)
                    noise = torch.randn_like(data) * self.noise_std
                    noisy_data = data + noise
                    denoised = self.model(noisy_data)
                    loss = self.compute_loss(denoised, data)
                    mse = self.criterion(denoised, data)
                    total_mse += mse.item()
                    total_loss += loss.item()
                print(f"Valid Epoch {epoch+1}, Valid Loss: {total_loss / len(dataloaders['valid']):.6f}")
                print(f"Angular Variance: {self.get_angular_variance()}")
                print(f"Valid MSE: {total_mse / len(dataloaders['valid'])}")
                if wandb.run:
                    wandb.log({"valid_loss": total_loss / len(dataloaders["valid"]), "angular_variance": self.get_angular_variance(), "valid_mse": total_mse / len(dataloaders["valid"])})

    def sample(self, num_samples=10):
        samples = torch.randn(num_samples, 40, device=device)  # Start from pure noise
        for _ in range(self.timesteps):
            samples = self.model(samples)
        return samples.view(num_samples, 20, 2).cpu()  #zz

    def visualize_samples(self, num_samples=5, filename="hexagons"):
        samples = self.sample(num_samples).detach().numpy()
        plt.figure(figsize=(6, 6))
        for i, sample in enumerate(samples):
            plt.scatter(sample[:, 0], sample[:, 1], label=f"Sample {i+1}")
        plt.legend()
        plt.title("Generated Hexagonal Point Clouds")
        plt.savefig(f"images/{filename}.png")
    
    def get_angular_variance(self, num_samples=1000):
        samples = self.sample(num_samples).detach().numpy()
        total_variance = 0
        for sample in samples:
            center_of_mass = np.mean(sample, axis=0)
            centered_sample = sample - center_of_mass
            angles = np.arctan2(centered_sample[:, 1], centered_sample[:, 0])
            angles = np.sort(angles)
            angle_differences = np.diff(np.concatenate((angles, [angles[0] + 2 * np.pi])))
            variance = np.var(angle_differences)
            radial_variance = np.var(np.linalg.norm(centered_sample, axis=1))
            total_variance += variance + radial_variance
        return np.log(total_variance / num_samples)

    def compute_energy_loss(self, denoised, ground_truth):
        edges = self.get_energy_edges(ground_truth)
        coeffs = self.get_energy_coeffs(ground_truth)
        dist_denoised = self.get_pairwise_distances(denoised)
        dist_ground_truth = self.get_pairwise_distances(ground_truth)
        error_energy = 0.5 * edges * coeffs * (dist_denoised - dist_ground_truth) ** 2 
        return error_energy.mean((1,2)) # sum in non-batch dimensions

    def get_pairwise_distances(self, points):
        diffs = points.unsqueeze(2) - points.unsqueeze(1)  # (B, 6, 6, 2)
        distances = torch.norm(diffs, dim=-1)  # (B, 6, 6)
        return distances

    def get_energy_edges(self, points):
        batch_size, num_points, _ = points.shape
        return torch.ones(batch_size, num_points, num_points, device=device) - torch.eye(num_points, device=device)

    def get_energy_coeffs(self, points):
        if self.coeff_mode == "constant":
            return torch.ones_like(self.get_energy_edges(points))
        elif self.coeff_mode == "exp_dist":
            distances = self.get_pairwise_distances(points)
            coeffs = torch.exp(-distances)
            return 2.6*coeffs