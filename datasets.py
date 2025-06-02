# datasets.py
from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

# 9. Define the DataLoader Function

def train_test_split_loader(
    batch_size:  int,
    num_workers: int  = 4,
    pin_memory:  bool = True,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
)-> Tuple[DataLoader, ...]:
    """
    Returns SIX DataLoaders:
      Task-1 (digits 0-3): train1, val1, test1
      Task-2 (digits 4-6): train2, val2, test2
      Task-3 (digits 7-9): train3, val3, test3
    Each subset is split train/val/test in the ratio
      (1 − val_frac − test_frac)  :  val_frac  :  test_frac.
    """
    assert 0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # --- load MNIST --------------------------------------------------------
    full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    targets = full.targets

    idx_t1 = ((targets >= 0) & (targets <= 3)).nonzero(as_tuple=True)[0]  # 0,1,2,3
    idx_t2 = ((targets >= 4) & (targets <= 6)).nonzero(as_tuple=True)[0]  # 4,5,6
    idx_t3 = ((targets >= 7) & (targets <= 9)).nonzero(as_tuple=True)[0]  # 7,8,9

    ds1, ds2, ds3 = map(Subset, (full, full, full), (idx_t1, idx_t2, idx_t3))

    # --- helper to split one subset into train/val/test --------------------
    def split(ds):
        n        = len(ds)
        n_test   = int(n * test_frac)
        n_val    = int(n * val_frac)
        n_train  = n - n_val - n_test
        return random_split(ds, [n_train, n_val, n_test])

    train1_ds, val1_ds, test1_ds = split(ds1)
    train2_ds, val2_ds, test2_ds = split(ds2)
    train3_ds, val3_ds, test3_ds = split(ds3)

    # --- DataLoaders -------------------------------------------------------
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    def loaders(tr, va, te):
        return (DataLoader(tr, shuffle=True,  **loader_args),
                DataLoader(va, shuffle=False, **loader_args),
                DataLoader(te, shuffle=False, **loader_args))

    return (*loaders(train1_ds, val1_ds, test1_ds),
            *loaders(train2_ds, val2_ds, test2_ds),
            *loaders(train3_ds, val3_ds, test3_ds))