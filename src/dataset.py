"""
PyTorch Dataset for FrozenLake world-model transitions.
Returns (obs_tensor, action_onehot, next_obs_tensor) tuples.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


DATA_PATH = "/data/koe/ECE285-Final/data/transitions.npz"


class TransitionDataset(Dataset):
    """
    Dataset of (o_t, a_t, o_{t+1}) transition tuples.

    Args:
        split: "train", "val", or "test"
        data_path: path to transitions.npz
        augment: apply random horizontal/vertical flip (only for train)
    """

    def __init__(self, split="train", data_path=DATA_PATH, augment=False):
        data = np.load(data_path)
        mask = data[f"{split}_mask"]

        self.obs      = data["obs"][mask]        # (N, 64, 64, 3) uint8
        self.next_obs = data["next_obs"][mask]
        self.actions  = data["actions"][mask].astype(np.int64)
        self.augment  = augment and (split == "train")

    def __len__(self):
        return len(self.obs)

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert uint8 HWC numpy image to float CHW tensor in [-1, 1]."""
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        return t

    def __getitem__(self, idx):
        obs      = self._to_tensor(self.obs[idx])       # (3, 64, 64)
        next_obs = self._to_tensor(self.next_obs[idx])
        action   = torch.tensor(self.actions[idx], dtype=torch.long)
        return obs, action, next_obs


def get_loaders(batch_size=64, data_path=DATA_PATH, num_workers=2):
    """Return train, val, test DataLoaders."""
    train_ds = TransitionDataset("train", data_path, augment=False)
    val_ds   = TransitionDataset("val",   data_path, augment=False)
    test_ds  = TransitionDataset("test",  data_path, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders(batch_size=32)
    obs, act, nxt = next(iter(train_loader))
    print(f"obs shape:      {obs.shape},  range [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"action shape:   {act.shape},  sample: {act[:8]}")
    print(f"next_obs shape: {nxt.shape}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")
    print(f"Test  batches: {len(test_loader)}")
