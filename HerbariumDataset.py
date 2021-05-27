from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class HerbariumDataset(Dataset):
    def __init__(self, annotations_file, image_metadata_file, img_dir,
                 transform=None):
        self.annotations = self.__get_annotations(annotations_file=annotations_file)
        self.image_metadata = pd.read_csv(image_metadata_file).set_index('id')
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __get_annotations(self, annotations_file):
        annotations = pd.read_csv(annotations_file).set_index('image_id')
        cur_ids = np.sort(annotations['category_id'].unique())
        new_ids = range(annotations.shape[0])
        map = {cur_ids[i]: new_ids[i] for i in range(len(cur_ids))}
        annotations['category_id'] = annotations['category_id'].map(map)
        return annotations

    def __len__(self):
        return len(self.image_metadata)

    def __getitem__(self, idx):
        image_id = self.image_metadata.iloc[idx].name
        filename = self.image_metadata.loc[image_id, 'file_name']
        img_path = self.img_dir / filename
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.annotations.loc[image_id, 'category_id']
        if self.transform:
            image = self.transform(image)
        sample = {"image": image, "label": label}
        return sample


if __name__ == '__main__':
    d = HerbariumDataset(annotations_file='data/annotations.csv',
                         image_metadata_file='data/images.csv', img_dir='data')
    print(d[0])
