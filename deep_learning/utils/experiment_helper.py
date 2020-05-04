import os


class ExperimentHelper(object):
  def __init__(self, work_dir='./', exp_name='default'):
    work_dir = os.path.expanduser(work_dir)

    self._work_dir = work_dir
    self._exp_root = os.path.join(work_dir, exp_name)
    self._dirs = {
      'checkpoints': os.path.join(work_dir, exp_name, 'checkpoints'),
      'tb_logs': os.path.join(work_dir, 'runs', exp_name),
    }

    for _, dirname in self._dirs.items():
      if not os.path.isdir(dirname):
        os.makedirs(dirname, 0o775)

  @property
  def exp_root(self):
    return self._exp_root

  @property
  def checkpoints(self):
    return self._dirs['checkpoints']

  @property
  def tb_logs(self):
    return self._dirs['tb_logs']
