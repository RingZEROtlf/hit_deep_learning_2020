import argparse
import yaml
import copy
import sys

import torch
import torchvision
from tensorboardX import SummaryWriter

from deep_learning.utils.import_helper import import_helper


class Interface(object):
  def __init__(self, config):
    self.config = copy.deepcopy(config)

    # check device
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build dataloaders
    self._build_dataloaders()

    # build model
    self.model = self.build_model().to(self.device)

    # build loss and optimizer
    self.criterion = torch.nn.CrossEntropyLoss()
    self._build_optimizer()

  def _build_dataset(self, config):
    cfg = copy.deepcopy(config)
    transforms = []
    for t in cfg['transforms']:
      if t['turn_on']:
        transforms.append(import_helper(t['type'], t.get('kwargs', {})))
    cfg['kwargs']['transform'] = torchvision.transforms.Compose(transforms)
    return import_helper(cfg['type'], cfg.get('kwargs', {}))

  def _build_dataloaders(self):
    config = copy.deepcopy(self.config['datasets'])

    # build train dataloader
    cfg_train = config['train']
    self.train_loader = torch.utils.data.DataLoader(
      dataset=self._build_dataset(cfg_train),
      batch_size=cfg_train['batch_size'],
      shuffle=True)

    # build test dataloader
    cfg_test = config['test']
    self.test_loader = torch.utils.data.DataLoader(
      dataset=self._build_dataset(cfg_test),
      batch_size=cfg_test['batch_size'],
      shuffle=False)

  def build_model(self):
    config = copy.deepcopy(self.config['net'])
    return import_helper(config['type'], config.get('kwargs', {}))

  def _build_optimizer(self):
    config = copy.deepcopy(self.config['trainer']['optimizer'])
    config.get('kwargs', {})['params'] = self.model.parameters()
    self.optimizer = import_helper(config['type'], config.get('kwargs', {}))

  def train(self):
    cfg_trainer = copy.deepcopy(self.config['trainer'])

    # begin train
    self.model.train()

    iteration = 0
    for epoch in range(cfg_trainer['max_epoch']):
      for i, (images, labels) in enumerate(self.train_loader):
        print(i)
        sys.stdout.flush()
        iteration += 1
        images = images.to(self.device)
        labels = labels.to(self.device)

        # forward
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        # backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (iteration + 1) % cfg_trainer['print_freq'] == 0:
          print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
            epoch+1, cfg_trainer['max_epoch'], 
            i+1, len(self.train_loader), 
            loss.item()))
      if (epoch + 1) % cfg_trainer['test_freq'] == 0:
        accuracy = self.evaluate()
        print('Epoch [{}/{}], Accuracy: {}%'.format(
          epoch+1, cfg_trainer['max_epoch'],
          accuracy))

  def evaluate(self):
    self.model.eval()
    with torch.no_grad():
      correct, total = 0, 0
      for images, labels in self.test_loader:
        images = images.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    self.model.train()
    return 100 * correct / total


def main():
  # parse arguments
  parser = argparse.ArgumentParser(description='PRDL 2020 Labs for HIT')
  parser.add_argument('--config')

  args = parser.parse_args()

  with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  interface = Interface(config)
  interface.train()


if __name__ == '__main__':
  main()
