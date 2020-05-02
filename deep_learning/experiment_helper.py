import os


class ExperimentHelper(object):
  def __init__(self, exp_root):
    self._exp_root = exp_root
    self.folders = {
      'checkpoints': os.path.join(exp_root, 'checkpoints'),
      'tb_logs': os.path.join(exp_root, 'tb_logs'),
    }

    for _, dirname in self.folders.items():
      if not os.path.isdir(dirname):
        os.makedirs(dirname, 0o775)

  @property
  def exp_root(self):
    return self._exp_root

  @property
  def checkpoints(self):
    return self.folders['checkpoints']

  @property
  def tb_logs(self):
    return self.folders['tb_logs']
