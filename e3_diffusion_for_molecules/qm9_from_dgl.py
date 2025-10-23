import torch
from torch.utils.data import DataLoader, Dataset
from dgl.data import QM9EdgeDataset
from torch.nn.utils.rnn import pad_sequence


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset

    Returns
    -------
    props : Pytorch tensor
        The dataset with only the retained information.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


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
    def __init__(self, label_keys=None):
        # Initialize the QM9EdgeDataset
        self.dataset = QM9EdgeDataset(label_keys=label_keys)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get the graph and labels
        graph, labels = self.dataset[idx]
        
        # Extract fields to match desired keys
        num_atoms = graph.num_nodes()
        # charges = graph.ndata['attr'][:, 0]  # Example: atomic charges (assuming stored in the 1st dim of `attr`)
        positions = graph.ndata['pos']  # 3D coordinates of atoms
        one_hot = graph.ndata['attr'][:, 1:]  # Example: one-hot encoding of atom type
                
        # Construct the dictionary
        data_dict = {
            "num_atoms": num_atoms,
            "charges": None,
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