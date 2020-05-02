import argparse
import logging
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from datasets import Cifar10Dataset
from models import AlexNet
from experiment_helper import ExperimentHelper

# build logger
logger = logging.getLogger('lab2')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# check device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
  # parse arguments
  parser = argparse.ArgumentParser(description='hit deep learning 2020')
  parser.add_argument('--dataset-root', default='~/dataset_repos')
  parser.add_argument('--exp-root', default='exp-default')
  parser.add_argument('--num-classes', type=int, default=10)
  parser.add_argument('--num-epochs', type=int, default=20)
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument('--learning-rate', type=float, default=0.001)
  parser.add_argument('--print-freq', type=int, default=100)
  parser.add_argument('--eval-freq', type=int, default=5)

  args = parser.parse_args()

  # build datasets
  train_dataset = Cifar10Dataset(args.dataset_root, 
                                 train=True,
                                 transform=transforms.ToTensor())
  
  eval_dataset = Cifar10Dataset(args.dataset_root,
                                train=False,
                                transform=transforms.ToTensor())

  # build dataloaders
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

  eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)

  # build model and tensorboard logger
  model = AlexNet(args.num_classes).to(device)

  # build loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

  # train and eval
  logger.info('{}'.format(args))
  train(model, train_loader, eval_loader, criterion, optimizer, args)


def train(model, train_loader, eval_loader, criterion, optimizer, args):
  # build experiment environment
  exp_helper = ExperimentHelper(args.exp_root)

  # build tensorboard logger
  tb_logger = SummaryWriter(exp_helper.tb_logs)

  # begin train
  model.train()

  iteration = 0
  total_steps = len(train_loader)
  best_accuracy = 0
  for epoch in range(args.num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      print(i)
      iteration += 1
      images = images.to(device)
      labels = labels.to(device)

      # forward
      outputs = model(images)
      loss = criterion(outputs, labels)

      # backward and optimize
      optimizer.zero_grad()
      loss.backward()
      tb_logger.add_scalar('loss', loss.item(), iteration)
      optimizer.step()

      if (i + 1) % args.print_freq == 0:
        logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, args.num_epochs, i+1, total_steps, loss.item()))
    if (epoch+1) % args.eval_freq == 0:
      accuracy = eval(model, eval_loader)

      # check best accuracy
      if best_accuracy < accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(),
                   os.path.join(exp_helper.exp_root, 'best_model.ckpt'))

      # save checkpoint
      logger.info('Epoch [{}/{}], Accuracy: {}%'.format(epoch+1, args.num_epochs, accuracy))
      torch.save(model.state_dict(),
                 os.path.join(exp_helper.checkpoints, 
                              'model_epoch_{:04d}.ckpt'.format(epoch+1)))
      tb_logger.add_scalar('accuracy', accuracy, epoch+1)


def eval(model, eval_loader):
  model.eval()
  with torch.no_grad():
    correct, total = 0, 0
    for images, labels in eval_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  model.train()
  return 100 * correct / total


if __name__ == '__main__':
  main()
