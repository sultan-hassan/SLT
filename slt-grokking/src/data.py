"""
Modular arithmetic dataset: predict (a + b) mod p.
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ModularAdditionDataset(Dataset):
    """All ordered pairs (a, b) in [0,p)^2 with label (a+b) mod p."""

    def __init__(self, p: int):
        self.p = p
        pairs = [(a, b) for a in range(p) for b in range(p)]
        self.a = torch.tensor([x[0] for x in pairs], dtype=torch.long)
        self.b = torch.tensor([x[1] for x in pairs], dtype=torch.long)
        self.c = (self.a + self.b) % p

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return self.a[idx], self.b[idx], self.c[idx]


def make_dataloaders(
    p: int = 97,
    train_frac: float = 0.5,
    batch_size: int = 128,
    seed: int = 42,
):
    """
    Returns (train_loader, test_loader, full_loader).
    full_loader iterates the entire dataset — used for stable LLC baseline.
    """
    dataset = ModularAdditionDataset(p)
    n_train = int(len(dataset) * train_frac)
    n_test = len(dataset) - n_train

    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              drop_last=False, pin_memory=False)
    full_loader  = DataLoader(dataset,  batch_size=batch_size, shuffle=True,
                              drop_last=False, pin_memory=False)

    return train_loader, test_loader, full_loader
