import os
import shutil

from deep_learning.commands.subcommand import Subcommand
from deep_learning.task_helper import TaskHelper
from deep_learning.utils.misc_helper import get_root_dir


class InitCommand(Subcommand):
  def add_subparser(self, name, parser):
    sub_parser = parser.add_parser(
      name, description='subcommand for init',
      help='initialize a lab directory')

    sub_parser.add_argument('lab_name', help='name of lab directory')

    sub_parser.set_defaults(run=_main)

    return sub_parser


def _main(args):
  shutil.copytree(os.path.join(get_root_dir(), 'configs'), 
                  os.path.join(args.lab_name, 'configs'))
