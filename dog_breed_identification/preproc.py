import os
import sys

import numpy as np
import pandas as pd
from google.colab import drive, files
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

drive.mount('/content/drive')
sys.path.append('/content/drive/My Drive')

df = pd.read_csv('/content/drive/My Drive/dog_breed/labels.csv')
top10 = df['breed'].value_counts()[:10].index
df.query("breed in @top10", inplace=True)
class_num = df['breed'].nunique()
classes =  df['breed'].unique()

encoder = LabelEncoder()
encoder.fit(df['breed'])
df['label'] = encoder.transform(df['breed'])
_, idx_test = train_test_split(df.index, stratify=df['label'], test_size=0.2, random_state=10)
df['set'] = 'train'
df.loc[df.index.isin(idx_test), 'set'] = 'test'

def get_path(rel_path, common_path='/content/drive/My Drive/dog_breed/train'):
    return os.path.join(common_path, rel_path)

paths = {}
labels = {}
for fold in ['train', 'test']:
    paths[fold] = df.query("set == @fold")['id'].apply(get_path).values + '.jpg'
    labels[fold] = df.query("set == @fold")['label'].values

train_mean = np.array([0.485, 0.456, 0.406])
train_std = np.array([0.229, 0.224, 0.225])

transform = {
    "train": transforms.Compose(
        [
            transforms.Resize(300),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(300),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ]
    ),
    "reverse": transforms.Compose(
        [
         transforms.Normalize(-train_mean, 1/train_std), # clarify about mean normalization
         transforms.ToPILImage(),
        ]
    ),
}

class Dogs(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, sample):
        image = Image.open(self.img_paths[sample])
        label = self.labels[sample]

        if self.transform:
            image = self.transform(image)

        return image, label

dataset = {
    sets: Dogs(paths[sets], labels[sets], transform[sets]) for sets in ["train", "test"]
}
loader = {
    'train' : DataLoader(dataset['train'], batch_size=64, shuffle=True, drop_last=True, pin_memory=True),
    'test' : DataLoader(dataset['test'], batch_size=256, shuffle=False),
}
