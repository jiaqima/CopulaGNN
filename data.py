import os

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def load_data(dataset="cora"):
    # Load data.
    path = os.path.join("data", dataset)
    assert dataset in ["cora", "pubmed", "citeseer"]
    dataset = Planetoid(
        root=path, name=dataset, transform=T.NormalizeFeatures())

    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.valid_mask = data.val_mask
    return data
