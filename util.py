import os
from pathlib import Path

import pandas as pd
import numpy as np


def split_image_metadata(path: str, valid_frac):
    image_metadata = pd.read_csv(path)
    indices = list(range(image_metadata.shape[0]))
    np.random.shuffle(indices)
    split = int(np.floor(valid_frac * len(indices)))
    train_idx, valid_idx = indices[split:], indices[:split]

    train = image_metadata.iloc[train_idx]
    valid = image_metadata.iloc[valid_idx]

    return train, valid
