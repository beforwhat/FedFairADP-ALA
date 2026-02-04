# data/dataset_factory.py

import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
import numpy as np


class FEMNISTDataset(Dataset):
    """Load FEMNIST from LEAF preprocessed format (all_data/train/ and all_data/test/)."""
    def __init__(self, root: str, train: bool = True, transform=None):
        self.transform = transform
        self.data = []
        self.targets = []
        self.writer_ids = []  # For natural partitioning by writer

        # Determine split directory
        split_name = "train" if train else "test"
        data_root = Path(root) / "all_data" / split_name

        if not data_root.exists():
            raise FileNotFoundError(
                f"FEMNIST LEAF '{split_name}' data not found at {data_root}. "
                "Please preprocess using LEAF with -iu flag: "
                "./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -iu"
            )

        json_files = sorted(data_root.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files in {data_root}")

        for json_file in json_files:
            with open(json_file, "r") as f:
                raw = json.load(f)
            for user_id, user_data in raw["user_data"].items():
                x = user_data["x"]
                y = user_data["y"]
                # LEAF FEMNIST: x is list of 784 floats, y is int
                for img, label in zip(x, y):
                    img = np.array(img, dtype=np.float32).reshape(28, 28)
                    self.data.append(img)
                    self.targets.append(int(label))
                    self.writer_ids.append(user_id)  # Keep original writer ID

        if not self.data:
            raise ValueError(f"No data loaded from {data_root}")

        self.data = np.stack(self.data)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            from PIL import Image
            img = Image.fromarray(img.astype(np.uint8), mode='L')
            img = self.transform(img)

        return img, target


def build_transform(transform_config: Dict[str, Any]):
    """Build torchvision transform from YAML config."""
    tfm_list = []
    for op_name, params in transform_config.items():
        if op_name == "ToTensor":
            tfm_list.append(transforms.ToTensor())
        elif op_name == "Normalize":
            tfm_list.append(transforms.Normalize(mean=params["mean"], std=params["std"]))
        elif op_name == "RandomHorizontalFlip":
            tfm_list.append(transforms.RandomHorizontalFlip())
        elif op_name == "RandomCrop":
            tfm_list.append(transforms.RandomCrop(size=params["size"], padding=params.get("padding", 4)))
        else:
            raise ValueError(f"Unsupported transform: {op_name}")
    return transforms.Compose(tfm_list)


def load_dataset(
    dataset_name: str,
    data_root: str,
    transform_train: Dict,
    transform_test: Dict,
    train_ratio: float = 1.0,
    **kwargs
) -> Tuple[Dataset, Dataset, int, Tuple[int, int, int]]:
    """
    Load dataset and return (trainset, testset, num_classes, input_shape).
    """
    data_root = Path(data_root) / dataset_name
    os.makedirs(data_root, exist_ok=True)

    train_transform = build_transform(transform_train)
    test_transform = build_transform(transform_test)

    if dataset_name == "cifar10":
        trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
        num_classes = 10
        input_shape = (3, 32, 32)

    elif dataset_name == "emnist":
        trainset = datasets.EMNIST(
            root=data_root, split="byclass", train=True, download=True, transform=train_transform
        )
        testset = datasets.EMNIST(
            root=data_root, split="byclass", train=False, download=True, transform=test_transform
        )
        num_classes = 62
        input_shape = (1, 28, 28)

    elif dataset_name == "svhn":
        # Load train + extra as training set
        train_part1 = datasets.SVHN(root=data_root, split="train", download=True, transform=train_transform)
        train_part2 = datasets.SVHN(root=data_root, split="extra", download=True, transform=train_transform)
        
        # Fix label: SVHN uses 1-10, where 10 means digit '0'
        train_part1.labels = [y if y != 10 else 0 for y in train_part1.labels]
        train_part2.labels = [y if y != 10 else 0 for y in train_part2.labels]
        trainset = torch.utils.data.ConcatDataset([train_part1, train_part2])
        
        testset = datasets.SVHN(root=data_root, split="test", download=True, transform=test_transform)
        testset.labels = [y if y != 10 else 0 for y in testset.labels]  # Also fix test labels
        
        num_classes = 10
        input_shape = (3, 32, 32)

    elif dataset_name == "femnist":
        trainset = FEMNISTDataset(root=data_root, train=True, transform=train_transform)
        testset = FEMNISTDataset(root=data_root, train=False, transform=test_transform)
        num_classes = 62
        input_shape = (1, 28, 28)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Optional: subsample training set (e.g., for faster debugging)
    if train_ratio < 1.0:
        n_train = len(trainset)
        indices = torch.randperm(n_train)[:int(n_train * train_ratio)]
        trainset = Subset(trainset, indices)

    return trainset, testset, num_classes, input_shape


def get_num_classes(dataset_name: str) -> int:
    """Quick lookup for number of classes."""
    mapping = {
        "cifar10": 10,
        "emnist": 62,
        "svhn": 10,
        "femnist": 62
    }
    return mapping[dataset_name]


def get_input_shape(dataset_name: str) -> Tuple[int, int, int]:
    """Quick lookup for input shape (C, H, W)."""
    mapping = {
        "cifar10": (3, 32, 32),
        "emnist": (1, 28, 28),
        "svhn": (3, 32, 32),
        "femnist": (1, 28, 28)
    }
    return mapping[dataset_name]