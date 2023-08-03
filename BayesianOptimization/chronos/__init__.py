import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.set_default_dtype(torch.float64)

elif torch.backends.mps.is_available():
    DEVICE = torch.device("cpu")
    torch.set_default_dtype(torch.float64)

else:
    DEVICE = torch.device("cpu")
    torch.set_default_dtype(torch.float64)

from .Optimizer import DiscreteGridOptimizer
from .SurrogateModels import GPSurrogate
from .EnsembleAcquisition import DQEnsembleAcquisition
from .AcquisitionFunctions import DQRandomSearch, DQProbabilityOfImprovement, DQExpectedImprovement, DQUpperConfidenceBound
