import numpy as np
from torch.utils.data import Dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate hexagon points
def generate_hexagon(scale=1.0, rotation=0.0, order=20):
    angles = np.linspace(0, 2 * np.pi, order + 1)[:-1]  # 6 points of a hexagon
    points = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * scale
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]
    ])
    return points @ rotation_matrix.T

# Custom dataset
class HexagonDataset(Dataset):
    def __init__(self, num_samples=10000, scale_range=[0.5, 1.5], aug_angle=0):
        self.data = []
        if aug_angle == -1:
            print("Augmentation angle not provided, using random rotation")
        else:
            print(f"Augmentation angle provided, using fixed rotation size {aug_angle}")


        for _ in range(num_samples):
            scale = np.random.uniform(scale_range[0], scale_range[1])  # Scale as input
            if aug_angle == -1:
                rotation = np.random.uniform(0, 2 * np.pi)
            else:
                rotation = np.random.uniform(-aug_angle, aug_angle)

            hexagon = generate_hexagon(scale = scale, rotation = rotation)
            self.data.append(torch.tensor(hexagon, dtype=torch.float32))
        self.data = torch.stack(self.data).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class HexagonRegressionDataset(Dataset):
    def __init__(self, num_samples=10000, scale_range=[0.5, 1.5], num_vertices = 20, aug_angle=0):
        self.inputs = []
        self.outputs = []
        if aug_angle == -1:
            print("Augmentation angle not provided, using random rotation")
        else:
            print(f"Augmentation angle provided, using fixed rotation size {aug_angle}")


        for i in range(num_samples):
            scale = np.random.uniform(scale_range[0], scale_range[1])  # Scale as input
            if aug_angle == -1:
                rotation = np.random.uniform(0, 2 * np.pi)
            else:
                rotation = np.random.uniform(0, aug_angle * 2 * np.pi)

            hexagon = generate_hexagon(scale = scale, rotation=rotation, order = num_vertices)  # Hexagon points as output
            
            self.inputs.append(torch.tensor([scale], dtype=torch.float32))
            self.outputs.append(torch.tensor(hexagon, dtype=torch.float32).flatten())  # Flatten (6,2) to (12,)

        self.inputs = torch.stack(self.inputs).to(device)
        self.outputs = torch.stack(self.outputs).to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]    
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = HexagonRegressionDataset(num_samples=10, scale_range=[0.5, 1.5], aug_angle=0.5, num_vertices=6)
    plt.figure(figsize=(8, 8))
    for i in range(len(dataset)):
        input, outputs = dataset[i]
        hexagon = outputs.view(-1, 2).cpu().numpy()
        plt.scatter(hexagon[:, 0], hexagon[:, 1], marker='o')

    plt.savefig("hexagons_aug_angle_check.png")