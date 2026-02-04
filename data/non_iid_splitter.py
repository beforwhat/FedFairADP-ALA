# data/non_iid_splitter.py

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset, Subset


def create_non_iid_partitions(
    dataset: Dataset,
    num_clients: int,
    non_iid_config: Dict[str, Any],
    seed: int = 42,
    save_path: str = None
) -> Dict[int, List[int]]:
    """
    Create client partitions based on non-IID strategy.
    
    Args:
        dataset: Full training dataset (with .targets attribute)
        num_clients: Number of clients to partition into
        non_iid_config: From YAML, e.g., {"strategy": "dirichlet", "alpha": 0.1}
        seed: Random seed for reproducibility
        save_path: Optional path to save partition indices (.pkl)
    
    Returns:
        client_id_to_indices: Dict mapping client_id -> list of sample indices
    """
    np.random.seed(seed)
    
    # Extract labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise AttributeError("Dataset must have 'targets' or 'labels' attribute")

    num_classes = len(np.unique(labels))
    client_id_to_indices = {i: [] for i in range(num_clients)}

    strategy = non_iid_config["strategy"]

    if strategy == "dirichlet":
        # Standard Dirichlet label distribution skew
        alpha = non_iid_config["alpha"]
        label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

        # Assign samples to clients
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = label_distribution[k]
            proportions = proportions / proportions.sum()  # normalize
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            client_idx_splits = np.split(idx_k, proportions)
            for i, split in enumerate(client_idx_splits):
                client_id_to_indices[i].extend(split.tolist())

    elif strategy == "natural_by_writer":
        # For FEMNIST: each writer is a client (pre-grouped in dataset)
        # Assumes dataset is FEMNISTDataset with per-writer grouping
        if not hasattr(dataset, 'writer_ids'):
            raise ValueError("Dataset must have 'writer_ids' for natural_by_writer")
        
        writer_ids = np.array(dataset.writer_ids)
        unique_writers = np.unique(writer_ids)
        
        if len(unique_writers) < num_clients:
            raise ValueError(f"Only {len(unique_writers)} writers available, but {num_clients} requested")
        
        # Select top-k writers by sample count (optional)
        if non_iid_config.get("min_samples_per_client", 0) > 0:
            valid_writers = []
            for w in unique_writers:
                if np.sum(writer_ids == w) >= non_iid_config["min_samples_per_client"]:
                    valid_writers.append(w)
            unique_writers = np.array(valid_writers)
        
        selected_writers = unique_writers[:num_clients]
        for i, writer in enumerate(selected_writers):
            idx = np.where(writer_ids == writer)[0]
            client_id_to_indices[i] = idx.tolist()

    elif strategy == "pathological":
        # Each client gets at most `n_shards` classes
        n_shards = non_iid_config["n_shards_per_client"]
        shard_size = len(dataset) // (num_clients * n_shards)
        idx_shard = [i for i in range(len(dataset))]
        np.random.shuffle(idx_shard)

        for i in range(num_clients):
            shards = []
            for _ in range(n_shards):
                if idx_shard:
                    shards.extend(idx_shard[:shard_size])
                    idx_shard = idx_shard[shard_size:]
            client_id_to_indices[i] = shards

    else:
        raise ValueError(f"Unsupported non-IID strategy: {strategy}")

    # Filter empty clients
    client_id_to_indices = {k: v for k, v in client_id_to_indices.items() if len(v) > 0}

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(client_id_to_indices, f)
        print(f"Saved partitions to {save_path}")

    return client_id_to_indices


def load_partitions(partition_path: str) -> Dict[int, List[int]]:
    """Load pre-saved client partitions."""
    with open(partition_path, 'rb') as f:
        return pickle.load(f)