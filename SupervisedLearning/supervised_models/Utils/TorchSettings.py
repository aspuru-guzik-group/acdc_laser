import torch
from torch import Tensor
from torch import nn


if torch.cuda.is_available():
    DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    torch.set_default_dtype(torch.float64)

elif torch.backends.mps.is_available():
    DEVICES = [torch.device("cpu")]
    torch.set_default_dtype(torch.float64)

else:
    DEVICES = [torch.device("cpu")]
    torch.set_default_dtype(torch.float64)