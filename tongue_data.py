# tongue_data.py - Dataloader and image augmentation functions.
# Author: hys
# Email: 1350460443@qq.com

from os.path import join, basename
from glob import glob
import random

import numpy as np
import torch
import cv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import KFold

seed = 18
random.seed(seed)

# Light augmentation
def light(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],),
        ToTensorV2()

    ])

# Strong augmentation
# Details in https://github.com/albumentations-team/albumentations
def strong(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.CLAHE(p=1),
        A.ShiftScaleRotate(p=1),
        A.RandomBrightness(p=1),
        A.Blur(blur_limit=11, p=1),
        A.GaussNoise(var_limit=(100, 150), p=1),
        A.HorizontalFlip(p=1),
        A.Cutout(p=1),
        A.RandomRotate90(p=1),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],),
        ToTensorV2()

    ])

# Use when validation, no augmentation, only load
def get_valid_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],),
        ToTensorV2()
    ])

class TongueDataset(Dataset):

    def __init__(self, input_dir, img_ids, num_classes, transforms, expand=True):
        super().__init__()
        self.img_dir = join(input_dir, 'images')
        self.label_dir = join(input_dir, 'labels')
        self.img_ids = img_ids
        self.num_classes = num_classes
        self.transforms = transforms
        self.expand = expand
        ''' 'expand' decides whether to expand the original dataset，
            which doubles the data set, imposes strong augmentation on
            half and light augmentation on the other.
        '''

        if self.expand:
            self._expand_data()

    # Double the data set
    def _expand_data(self):
        self.img_ids = [x+'_aug' for x in self.img_ids] + self.img_ids
        random.shuffle(self.img_ids)

    # read imgs and labels
    def _imread(self, img_id):
        img = cv2.imread(join(self.img_dir, img_id+'.jpg'))  # load image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        with open(join(self.label_dir, img_id+'.txt'), 'r') as f:  # load labels from file
            l = np.array([x.split() for x in f.read().splitlines()])
            labels = l[:,0].astype(np.int).tolist()

        # transform labels to one-hot encoding
        onehot = np.zeros(self.num_classes)
        for i in labels:
            if i < self.num_classes:  # deprecate the labels outnumbered num_classes
                onehot[i] = 1.0

        return img, onehot

    def __getitem__(self, index: int):
        if self.expand:
            img_id = self.img_ids[index]
            is_expand = True if img_id.endswith('_aug') else False
            img, labels = self._imread(img_id[:-4]) if is_expand else self._imread(img_id)
            augmented = self.transforms['strong'](image=img) if is_expand else self.transforms['light'](image=img)
            img = augmented['image']
            return img, torch.tensor(labels)

        else:
            img_id = self.img_ids[index]
            img, labels = self._imread(img_id)
            augmented = self.transforms(image=img)
            img = augmented['image']
            return img, torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.img_ids)


def get_loaders(input_dir, num_classes, img_size, batch_size, num_workers, fold=0):
    img_ids = [basename(x).split('.')[0] for x in glob(join(input_dir, 'images', '*.jpg'))]  # load image filename to a list
    
    # random.shuffle(img_ids)  # shuffle the list
    # img_counts = len(img_ids)
    # train_ids = img_ids[:-int(img_counts*0.2)]  # 80% of all used to train, the other val
    # val_ids = img_ids[-int(img_counts*0.2):]

    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    folds = [[train.tolost(), val.tolist()] for train, val in kf.split(img_ids)]
    train_ids, val_ids = folds[fold]

    train_set = TongueDataset(
        input_dir, 
        train_ids, 
        num_classes, 
        {
            'strong': strong(img_size),
            'light': light(img_size)
        },
        expand = True
    )
    val_set = TongueDataset(
        input_dir, 
        val_ids, 
        num_classes, 
        get_valid_transform(img_size),
        expand = False
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader