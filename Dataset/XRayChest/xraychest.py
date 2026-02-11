from torch.utils.data import Dataset
import kagglehub
import os

from pathlib import Path
from PIL import Image
import torchvision.transforms as T

class XRayChestDataset(Dataset):
    def __init__(self, split="train", img_size=256, transform=None):
        root = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")) / "chest_xray" / "chest_xray"
        split_dir = root / split
        samples = []

        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir() or cls_dir.name.startswith("."):
                continue
            for img_path in cls_dir.glob("*.jpeg"):
                samples.append(img_path)

        self.samples = samples
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ]) if transform is None else transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img


if __name__ == "__main__":
    # Resize any input to 1024x1024, then convert to tensor in [0, 1].
    transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
    ])

    dataset = XRayChestDataset(transform=transform)
    print(f"Total samples: {len(dataset)}")

    sample = dataset[10]
    print(sample.shape)

    # Convert tensor back to PIL for saving/debugging.
    # pil_img = T.ToPILImage()(sample)
    # out_path = Path("./sample_10_resized.jpg")
    # pil_img.save(out_path)
    # print(f"Saved {out_path.resolve()}")