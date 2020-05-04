import argparse

import yaml

from deep_learning.commands import InitCommand, RunCommand


def main():
  # parse arguments
  parser = argparse.ArgumentParser(description='PRDL 2020 Labs for HIT')

  subparsers = parser.add_subparsers(title='subcommands')

  subcommands = {
    'init': InitCommand(),
    'run': RunCommand(),
  }
  
  for subname, subcommand in subcommands.items():
    subcommand.add_subparser(subname, subparsers)

  args = parser.parse_args()
  args.run(args)


if __name__ == '__main__':
  main()
