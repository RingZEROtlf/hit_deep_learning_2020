import yaml

from deep_learning.commands.subcommand import Subcommand
from deep_learning.task_helper import TaskHelper


class RunCommand(Subcommand):
  def add_subparser(self, name, parser):
    sub_parser = parser.add_parser(
      name, description='subcommand for run',
      help='run a task')

    sub_parser.add_argument('exp_name', help='name of current experiment')
    sub_parser.add_argument('--config', help='configuration file')

    sub_parser.set_defaults(run=_main)

    return sub_parser


def _main(args):
  with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  task_helper = TaskHelper(config, exp_name=args.exp_name)
  task_helper.get_summary()
