import copy
import os
import yaml

import torch
import torchscan
import torchvision
from tensorboardX import SummaryWriter

from deep_learning.utils.experiment_helper import ExperimentHelper
from deep_learning.utils.import_helper import import_helper
from deep_learning.utils.log_helper import default_logger


class TaskHelper(object):
  def __init__(self, config, work_dir='./', exp_name='default'):
    self.config = copy.deepcopy(config)

    # build logger
    self.exp_helper = ExperimentHelper(work_dir=work_dir, exp_name=exp_name)
    self.logger = default_logger

    # check device
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build dataloaders
    self._build_dataloaders()

    # build model
    self.model = self.build_model().to(self.device)

    # build loss and optimizer
    self.criterion = torch.nn.CrossEntropyLoss()
    self._build_optimizer()

  def get_summary(self):
    torchscan.summary(self.model, self.get_dummy_input()[0].shape[1:])

  def get_model(self):
    return self.model

  def get_dummy_input(self):
    input_shapes = self.config['net']['input_shapes']
    dummy_inputs = tuple(map(lambda x: torch.randn(*x), input_shapes))
    return dummy_inputs

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
    with open(os.path.join(self.exp_helper.exp_root, 'config.yaml'), 'w') as f:
      f.write(yaml.dump(self.config))
    cfg_trainer = copy.deepcopy(self.config['trainer'])
    summary_writer = SummaryWriter(self.exp_helper.tb_logs)

    # begin train
    self.model.train()

    iteration = 0
    best_accuracy = 0
    for epoch in range(cfg_trainer['max_epoch']):
      for i, (images, labels) in enumerate(self.train_loader):
        iteration += 1
        images = images.to(self.device)
        labels = labels.to(self.device)

        # forward
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        # backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        summary_writer.add_scalar('loss', loss.item(), iteration)
        self.optimizer.step()

        if iteration % cfg_trainer['print_freq'] == 0:
          self.logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            .format(epoch+1, cfg_trainer['max_epoch'], 
                    i+1, len(self.train_loader), 
                    loss.item()))
      if (epoch + 1) % cfg_trainer['test_freq'] == 0:
        accuracy = self.evaluate()
        self.logger.info('Epoch [{}/{}], Accuracy: {}%'
          .format(epoch+1, cfg_trainer['max_epoch'],
                  accuracy))
        # save the checkpoint
        torch.save(self.model.state_dict(), 
                   os.path.join(self.exp_helper.checkpoints, 
                                'model_epoch_{}.ckpt'.format(epoch+1)))
        # write to tensorboard
        summary_writer.add_scalar('accuracy', accuracy, epoch+1)
        # check best model
        if best_accuracy < accuracy:
          best_accuracy = accuracy
          torch.save(self.model.state_dict(), 
                    os.path.join(self.exp_helper.exp_root, 
                                 'best_model.ckpt'))
    summary_writer.close()

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
