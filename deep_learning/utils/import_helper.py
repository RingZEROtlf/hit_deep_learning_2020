import importlib


def import_helper(type, kwargs):
  module_name, cls_name = type.rsplit('.', 1)
  module = importlib.import_module(module_name)
  cls_ = getattr(module, cls_name)
  return cls_(**kwargs)
