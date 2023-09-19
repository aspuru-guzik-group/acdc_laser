import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(format="%(message)s", level=logging.ERROR)

from .SupervisedModel import SupervisedModel
from .Models import *
from .Encoders import *
