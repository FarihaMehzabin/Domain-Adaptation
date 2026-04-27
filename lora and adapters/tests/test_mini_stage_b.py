import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path("/workspace")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_nih_2k_baseline import LABELS, NIH2KDataset, build_model, compute_pos_weight


TRAIN_CSV = ROOT / "manifests" / "nih_dev_2k_train.csv"


def load_train_dataframe() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV).head(4).copy()
    df["resolved_path"] = df["abs_path"]
    return df


def test_dataset_loads_one_item() -> None:
    dataset = NIH2KDataset(load_train_dataframe(), image_size=224)
    image, labels = dataset[0]
    assert image.shape == (3, 224, 224)
    assert labels.shape == (5,)
    assert labels.dtype == torch.float32


def test_label_vector_is_binary_float() -> None:
    dataset = NIH2KDataset(load_train_dataframe(), image_size=224)
    _, labels = dataset[0]
    values = labels.numpy()
    assert len(values) == len(LABELS)
    assert np.all(np.isin(values, [0.0, 1.0]))


def test_model_forward_shape_and_loss() -> None:
    dataset = NIH2KDataset(load_train_dataframe(), image_size=224)
    batch = torch.stack([dataset[i][0] for i in range(2)], dim=0)
    targets = torch.stack([dataset[i][1] for i in range(2)], dim=0)

    model, _ = build_model(prefer_pretrained=False)
    logits = model(batch)
    assert logits.shape == (2, len(LABELS))

    pos_weight = compute_pos_weight(load_train_dataframe())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = criterion(logits, targets)
    assert not torch.isnan(loss)
