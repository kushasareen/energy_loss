from torch_geometric.datasets import QM9
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class PreprocessQM9:
    def __init__(self, load_charges=True):
        self.load_charges = load_charges

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for the QM9 dataset.

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of Pytorch tensors
            The collated data.
        """
        # Stack tensors and pad where necessary
        batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

        # Determine which atoms to keep (non-zero charges)
        to_keep = (batch['charges'].sum(0) > 0)

        # Drop zero entries based on `to_keep`
        batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

        # Create atom mask
        atom_mask = batch['charges'] > 0
        batch['atom_mask'] = atom_mask

        # Obtain edge mask
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        # Mask diagonal (self-loops)
        diag_mask = ~torch.eye(n_nodes, dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask

        # Flatten edge mask
        batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        # Add dimension to charges if needed
        if self.load_charges:
            batch['charges'] = batch['charges'].unsqueeze(2)
        else:
            batch['charges'] = torch.zeros(0)

        return batch

class QM9CustomDataset(Dataset):
    def __init__(self, label_keys=None, data_path=''):
        # Initialize the QM9EdgeDataset
        self.dataset = QM9(root=f'{data_path}/datasets')
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get the graph and labels
        mol = self.dataset[idx]
        
        # Extract fields to match desired keys
        num_atoms = len(mol.z)
        charges = torch.tensor(mol.z, dtype=torch.float32)
        positions = mol.pos  # 3D coordinates of atoms
        one_hot = graph.ndata['attr'][:, 1:]  # Example: one-hot encoding of atom type
        labels = mol.y
                
        # Construct the dictionary
        data_dict = {
            "num_atoms": num_atoms,
            "charges": charges,
            "positions": positions,
            "index": idx,
            "A": labels[12],
            "B": labels[13],
            "C": labels[14],
            "mu": labels[0],
            "alpha": labels[1],
            "homo": labels[2],
            "lumo": labels[3],
            "gap": labels[4],
            "r2": labels[5],
            "zpve": labels[6],
            "U0": labels[7],
            "U": labels[8],
            "H": labels[9],
            "G": labels[10],
            "Cv": labels[11],
            # "omega1": omega1,
            # "zpve_thermo": zpve_thermo,
            # "U0_thermo": U0_thermo,
            # "U_thermo": U_thermo,
            # "H_thermo": labels[12],
            # "G_thermo": labels[13],
            # "Cv_thermo": labels[14],
            "one_hot": one_hot,
            "edges": graph.edges(),  # Add edges (source and destination nodes)
        }

        breakpoint()
        
        return data_dict

# # Define a custom collate function
# def collate_fn(batch):
#     # Batch each key individually
#     batched_data = {key: [] for key in batch[0].keys()}
    
#     for data in batch:
#         for key, value in data.items():
#             if isinstance(value, torch.Tensor):
#                 batched_data[key].append(value)
#             else:
#                 batched_data[key].append(torch.tensor(value) if isinstance(value, int) else value)
    
#     # Stack tensors where applicable
#     for key, value in batched_data.items():
#         if isinstance(value[0], torch.Tensor):
#             batched_data[key] = torch.stack(value) if value[0].ndim > 0 else torch.tensor(value)
    
#     return batched_data

# Instantiate the dataset and DataLoader
label_keys = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv", "A", "B", "C"]
dataset = QM9CustomDataset(label_keys=label_keys)

preprocessor = PreprocessQM9(load_charges=True)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=preprocessor.collate_fn,
)

# Iterate through DataLoader
for batch in dataloader:
    print(batch.keys())  # Display keys in the batch
    print(batch['charges'].shape)  # Check the shape of 'charges'
    break