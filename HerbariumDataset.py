from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class HerbariumDataset(Dataset):
    def __init__(self, annotations_file, image_metadata_file, img_dir,
                 transform=None, target_transform=None):
        self.annotations = pd.read_csv(annotations_file).set_index('image_id')
        self.image_metadata = pd.read_csv(image_metadata_file).set_index('id')
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_id = self.image_metadata.iloc[idx].name
        filename = self.image_metadata.loc[image_id, 'file_name']
        img_path = self.img_dir / filename
        image = Image.open(img_path)
        label = self.annotations.loc[image_id, 'category_id']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample


if __name__ == '__main__':
    d = HerbariumDataset(annotations_file='data/annotations.csv',
                         image_metadata_file='data/images.csv', img_dir='data')
    print(d[0])
