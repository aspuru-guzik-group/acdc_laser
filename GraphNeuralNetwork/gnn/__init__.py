import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("deepchem").setLevel(logging.CRITICAL)

from .MoleculeEncoder import MolFeaturizer
from .GNN import GNN
