import torch


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

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


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
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


class PreprocessQM9:
    def __init__(self, load_charges=True, k_regular=False):
        self.load_charges = load_charges
        if k_regular:
            self.k_regular = torch.load('rigidity/k_regular.pt') # Shape (27, 1000, 29, 29), index 0 is for n=2
        else:
            self.k_regular = None

    def add_trick(self, trick):
        self.tricks.append(trick)

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of Pytorch tensors
            The collated data.
        """
        batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

        to_keep = (batch['charges'].sum(0) > 0)

        if 'edges' in batch.keys():
            batch['edges'] = batch['edges'][:, :, to_keep]

        if 'sym_edges' in batch.keys():
            batch['sym_edges'] = batch['sym_edges'][:, :, to_keep]

        if self.k_regular is not None:
            # key is the number of atoms - 2
            keys = batch['num_atoms'] - 2
            # pick a random k_regular graph for each sample with the correct number of atoms
            random_idx = torch.randint(0, self.k_regular.size(0), (keys.size(0),))
            batch['k_regular'] = self.k_regular[keys, random_idx, :]
            # batch['k_regular'] = batch['k_regular'][:, random_idx, :, :]
            batch['k_regular'] = batch['k_regular'][:, :, to_keep]

        batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

        atom_mask = batch['charges'] > 0
        batch['atom_mask'] = atom_mask

        #Obtain edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask

        #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        if self.load_charges:
            batch['charges'] = batch['charges'].unsqueeze(2)
        else:
            batch['charges'] = torch.zeros(0)
        return batch
