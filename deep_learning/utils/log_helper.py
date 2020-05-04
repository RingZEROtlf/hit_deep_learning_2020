import logging
import sys

logs = set()

# LOGGER
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = '\033[0m'
COLOR_SEQ = '\033[1;%dm'

COLORS = {
  'WARNING': YELLOW,
  'INFO': WHITE,
  'DEBUG': BLUE,
  'CRITICAL': YELLOW,
  'ERROR': RED,
}

class ColoredFormatter(logging.Formatter):
  def __init__(self, msg, use_color=True):
    logging.Formatter.__init__(self, msg)
    self.use_color = use_color

  def format(self, record):
    msg = record.msg
    levelname = record.levelname
    if self.use_color and levelname in COLORS and COLORS[levelname] != WHITE:
      if isinstance(msg, str):
        msg_color = COLOR_SEQ % (30 + COLORS[levelname]) + msg + RESET_SEQ
        record.msg = msg_color
      levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
      record.levelname = levelname_color
    return logging.Formatter.format(self, record)


def init_log(name, level=logging.INFO):
  if (name, level) in logs:
    return

  logs.add((name, level))
  logger = logging.getLogger(name)
  logger.setLevel(level)

  handler = logging.StreamHandler(stream=sys.stdout)
  handler.setLevel(level)

  format_str = f'[%(asctime)s]\n%(message)s'
  formatter = ColoredFormatter(format_str)
  handler.setFormatter(formatter)

  logger.addHandler(handler)

  return logger


default_logger = init_log('global', logging.INFO)
