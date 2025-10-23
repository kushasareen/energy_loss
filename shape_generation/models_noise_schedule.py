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
class MLPDenoiser(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, time_dim=16):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.model = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t.view(-1, 1))  # Encode time
        x = torch.cat([x, t_emb], dim=-1)  # Concatenate time embedding with input
        return self.model(x)

# Noise schedules
def alpha_schedule(t, timesteps):
    return 1.0 - 0.5 * (t / timesteps)  # Example alpha schedule

def compute_alpha_bar(t, timesteps):
    alpha_values = [alpha_schedule(i, timesteps) for i in range(t + 1)]
    return np.prod(alpha_values)  # Cumulative product

class DiffusionModel:
    def __init__(self, args, denoiser=MLPDenoiser(), criterion=nn.MSELoss(), timesteps=100):
        self.timesteps = timesteps
        self.model = denoiser.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.criterion = criterion
        self.test_epochs = 10
        self.loss_mode = args.loss_mode  # "energy" or "mse"
        self.coeff_mode = args.coeff_mode
        self.alpha_bars = torch.tensor([compute_alpha_bar(t, timesteps) for t in range(timesteps)], device=device, dtype=torch.float32)
        self.args = args

    def train(self, dataloaders, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_grad_norm = 0
            for data in dataloaders["train"]:
                data = data.view(data.size(0), -1).to(device)  # Flatten (batch_size, 12)

                t = torch.randint(0, self.timesteps, (data.size(0),), device=device)  # Sample time step
                alpha_bar_t = self.alpha_bars[t].view(-1, 1)
                noise = torch.randn_like(data) * torch.sqrt(1 - alpha_bar_t)  # Apply noise schedule

                noisy_data = torch.sqrt(alpha_bar_t) * data + noise  # Variance-preserving scaling
                self.optimizer.zero_grad()
                denoised = self.model(noisy_data, t.float() / self.timesteps)  # Denoise

                loss = compute_loss(denoised, data, loss_mode=self.loss_mode, coeff_mode=self.coeff_mode, base_loss_mode=self.args.base_loss_mode)
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
            for data in dataloaders["valid"]:
                data = data.view(data.size(0), -1).to(device)  # Flatten (batch_size, 12)

                t = torch.randint(0, self.timesteps, (data.size(0),), device=device)
                alpha_bar_t = self.alpha_bars[t].view(-1, 1)
                noise = torch.randn_like(data) * torch.sqrt(1 - alpha_bar_t)

                noisy_data = torch.sqrt(alpha_bar_t) * data + noise
                denoised = self.model(noisy_data, t.float() / self.timesteps)

                loss = compute_loss(denoised, data, loss_mode=self.loss_mode, coeff_mode=self.coeff_mode, base_loss_mode=self.args.base_loss_mode)
                mse = self.criterion(denoised, data)
                total_mse += mse.item()
                total_loss += loss.item()

            print(f"Valid Loss: {total_loss / len(dataloaders['valid'])}")
            print(f"Valid MSE: {total_mse / len(dataloaders['valid'])}")
            diversity_metric, quality_metric, overall_metric, samples_grad = self.get_metrics()
            print(f"Diversity Metric: {diversity_metric:.6f}, Quality Metric: {quality_metric:.6f}, Overall Metric: {overall_metric:.6f}, Samples Grad Norm: {samples_grad:.6f}")
            if wandb.run:
                wandb.log({"Valid Loss": total_loss / len(dataloaders["valid"]),
                           "Valid MSE": total_mse / len(dataloaders["valid"]),
                           "Diversity Metric": diversity_metric,
                           "Quality Metric": quality_metric,
                           "Overall Metric": overall_metric,
                           "Samples Grad Norm": samples_grad})

    def sample(self, num_samples=10):
        samples = torch.randn(num_samples, 40, device=device)  # Start from pure noise
        for t in reversed(range(self.timesteps)):
            alpha_bar_t = self.alpha_bars[[t]].repeat(num_samples,1)
            ts = torch.full((num_samples,), t, device=device)
            denoised = self.model(samples, ts / self.timesteps)
            samples = torch.sqrt(alpha_bar_t) * denoised + torch.randn_like(samples) * torch.sqrt(1 - alpha_bar_t)
        return samples.view(num_samples, 20, 2).cpu()

    def visualize_samples(self, num_samples=5, filename="hexagons"):
        samples = self.sample(num_samples).detach().numpy()
        plt.figure(figsize=(6, 6))
        for i, sample in enumerate(samples):
            plt.scatter(sample[:, 0], sample[:, 1], label=f"Sample {i+1}")
        plt.legend()
        plt.title("Generated Hexagonal Point Clouds")
        plt.savefig(f"images/{filename}.png")

    def get_metrics(self, num_samples=1000): # lower is better
        samples = self.sample(num_samples)
        samples_grad = self.log_gradient_wrt_samples(samples)
        samples = samples.detach().numpy()
        center_of_mass = np.mean(samples, axis=1)
        centered_samples = samples - center_of_mass[:, None]
        total_variance = 0
        for centered_sample in centered_samples:
            angles = np.arctan2(centered_sample[:, 1], centered_sample[:, 0])
            angles = np.sort(angles)
            angle_differences = np.diff(np.concatenate((angles, [angles[0] + 2 * np.pi])))
            variance = np.var(angle_differences)
            radial_variance = np.var(np.linalg.norm(centered_sample, axis=1))
            total_variance += variance + radial_variance
                
                
        diversity_metric = -np.log(np.linalg.norm(samples, axis=2).mean(-1).var()) # lower is better
        quality_metric = np.log(total_variance / num_samples)
        overall_metric = diversity_metric + quality_metric
        return diversity_metric, quality_metric, overall_metric, samples_grad

    def log_gradient_wrt_samples(self, samples):
        # samples = samples.to(device)
        # samples.requires_grad = True
        # t = (torch.ones((samples.size(0),), device=device) * (self.timesteps - 1)).int()
        # alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        # noise = torch.randn_like(samples) * torch.sqrt(1 - alpha_bar_t)
        # noisy_data = torch.sqrt(alpha_bar_t) * samples + noise
        # noisy_data = noisy_data.view(-1, 40)
        # denoised = self.model(noisy_data, t.float() / self.timesteps)
        # samples = samples.view(-1, 40)
        # breakpoint()
        # loss = compute_loss(denoised, samples, loss_mode=self.loss_mode, coeff_mode=self.coeff_mode, base_loss_mode=self.args.base_loss_mode)
        # # loss = compute_loss(samples, samples, loss_mode=self.loss_mode, coeff_mode=self.coeff_mode, base_loss_mode=self.args.base_loss_mode)
        # # loss = Variable(loss, requires_grad=True)
        # loss.backward()
        # grad = samples.grad
        # breakpoint()
        # grad_norm = torch.norm(grad, dim=-1).mean()
        # breakpoint()
        # return grad_norm.item()

        return 0