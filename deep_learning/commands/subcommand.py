import abc


class Subcommand(abc.ABC):
  @abc.abstractmethod
  def add_subparser(self, name, parser):
    raise NotImplementedError
