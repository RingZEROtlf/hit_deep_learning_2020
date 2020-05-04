import os
import pickle

import numpy as np
import torch
from PIL import Image

__all__ = ['Cifar10Dataset']


class Cifar10Dataset(torch.utils.data.Dataset):
  base_folder = 'cifar-10-batches-py'
  train_list = ['data_batch_' + str(idx) for idx in range(1, 6)]
  test_list = ['test_batch']

  def __init__(self, root, train=True, transform=None):
    if isinstance(root, torch._six.string_classes):
      root = os.path.expanduser(root)
    self.root = root

    self.train = train # train or test

    self.transform = transform

    downloaded_list = self.train_list if self.train else self.test_list

    self.data = []
    self.targets = []

    # load the picked numpy arrays
    for filename in downloaded_list:
      filepath = os.path.join(self.root, self.base_folder, filename)
      with open(filepath, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        self.data.append(entry['data'])
        if 'labels' in entry:
          self.targets.extend(entry['labels'])
        else:
          self.targets.extend(entry['fine_labels'])

    self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
    self.data = self.data.transpose((0, 2, 3, 1)) # convert to HWC

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img, target = self.data[idx], self.targets[idx]

    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    return img, target
