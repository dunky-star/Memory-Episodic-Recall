# datasets.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

def train_test_split_loader(
    batch_size:  int,
    num_workers: int  = 4,
    pin_memory:  bool = True,
    test_frac:   float = 0.2
):
    """
    Returns four DataLoaders for Split‑MNIST with an 80/20 train‑test split in each task:
      - Task1 (digits0–4): train1, test1
      - Task2 (digits5–9): train2, test2
    """
    # Common MNIST transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Load full MNIST training set
    full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    targets = full.targets

    # Indices for the two splits
    idx1 = (targets < 5).nonzero(as_tuple=True)[0]
    idx2 = (targets >= 5).nonzero(as_tuple=True)[0]

    ds1 = Subset(full, idx1)  # digits 0–4
    ds2 = Subset(full, idx2)  # digits 5–9

    # 80/20 split for each subset
    def split(ds):
        n = len(ds)
        n_test = int(n * test_frac)
        n_train = n - n_test
        return random_split(ds, [n_train, n_test])

    train1_ds, test1_ds = split(ds1)
    train2_ds, test2_ds = split(ds2)

    # Build DataLoaders
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train1 = DataLoader(train1_ds, shuffle=True,  **loader_args)
    test1  = DataLoader(test1_ds,  shuffle=False, **loader_args)
    train2 = DataLoader(train2_ds, shuffle=True,  **loader_args)
    test2  = DataLoader(test2_ds,  shuffle=False, **loader_args)

    return train1, test1, train2, test2
