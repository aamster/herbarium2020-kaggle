import os
from pathlib import Path

import pandas as pd
import numpy as np


def split_image_metadata(path: str, train_frac):
    image_metadata = pd.read_csv(path)
    indices = list(range(image_metadata.shape[0]))
    np.random.shuffle(indices)
    split = int(np.floor(train_frac * len(indices)))
    midpoint = int((len(indices) - split) / 2)
    train_idx, valid_idx, test_idx = \
        indices[:split], indices[split:midpoint], indices[midpoint:]

    train = image_metadata.iloc[train_idx]
    valid = image_metadata.iloc[valid_idx]
    test = image_metadata.iloc[test_idx]

    return train, valid, test
