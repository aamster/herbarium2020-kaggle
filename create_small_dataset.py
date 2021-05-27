import cv2
import pandas as pd
import json
import os

import torchvision
from tqdm import tqdm
import numpy as np
import h5py

from pathlib import Path

from Transforms import CropBorder


def filter_top_categories(annotations, images):
    """
    Filters possible images to only top 1000 categories
    """
    keep_cats = annotations.groupby('category_id').size().sort_values(
        ascending=False)[:1000].index
    keep_ann = annotations[annotations['category_id'].isin(keep_cats)]
    keep_ann.to_csv('annotations.csv', index=False)

    keep_img_ids = keep_ann['image_id']
    keep_imgs = images[images['id'].isin(keep_img_ids)]
    keep_imgs.to_csv('images.csv', index=False)
    print(f'keep {keep_imgs.shape[0]} imgs')


def create_dataset(images, src, dst):
    for file in tqdm(images['file_name']):
        src_file = Path(src / file)
        dst_file = Path(dst / file)
        os.makedirs(dst_file.parent, exist_ok=True)
        os.system(f'cp {src_file} {dst_file}')


def create_dataset_h5py(images, src):
    dset = np.zeros((images.shape[0], 448, 314, 3))
    for i, file in tqdm(enumerate(images['file_name']), total=images.shape[0]):
        src_file = Path(src / file)
        img = cv2.imread(str(src_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transforms = torchvision.transforms.Compose([
            CropBorder(),
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((448, 314))
        ])
        img = transforms(img)
        img = np.array(img)
        dset[i] = img

    with h5py.File('/Users/adam.amster/herbarium-2020-fgvc7-small.h5',
                   'w') as f:
        f.create_dataset('data', data=dset)


def main():
    with open(
            '/Users/adam.amster/herbarium-2020-fgvc7/nybg2020/train/metadata'
            '.json',
            encoding='latin-1') as f:
        metadata = json.loads(f.read())

    annotations = pd.DataFrame(metadata['annotations'])
    images = pd.DataFrame(metadata['images'])

    filter_top_categories(annotations=annotations, images=images)

    src = Path('/Users/adam.amster/herbarium-2020-fgvc7/nybg2020/train/')

    dst = Path('/Users/adam.amster/herbarium-2020-fgvc7-small/nybg2020/train/')

    images = pd.read_csv('images.csv')
    create_dataset(images=images, src=src, dst=dst)
    # create_dataset_(images=images, src=src)


if __name__ == '__main__':
    main()