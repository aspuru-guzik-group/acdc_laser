# This is an example script to build a supervised learning model based on the experimental seed data to predict the
# experimental properties using a multitask Gaussian Process regressor (as shown in the paper).

from pathlib import Path
import pandas as pd
from gpytorch.kernels import RBFKernel
from supervised_models import MultitaskGaussianProcess
from supervised_models.Utils import IdentityScaler
from supervised_models.Utils import MultipleKFoldSplitter


PROJECT_ROOT = Path.cwd().parent

# Loads the observations and the GNN features for all experimental datapoints from the "seed dataset" as an example
observations = pd.read_csv(
    PROJECT_ROOT / "Data" / "experimental_seed_dataset" / "experimental_observations.csv",
    index_col=False,
    usecols=["Gain Cross Section (cm^2)", "Quantum Yield", "Emission Lifetime (ns)", "Spectral Gain Factor (cm^2 s)", "Emission Wavelength (nm)"]
)

gnn_features = pd.read_csv(
    PROJECT_ROOT / "Data" / "experimental_seed_dataset" / "representations" / "gnn_embeddings.csv",
    index_col=False,
    header=None
)

# Defines the model architecture and the train-test splitting settings
model_architecture: dict = {
    "kernel": RBFKernel,
    "learningrate": 0.05,
    "training_iterations": 2000,
    "feature_lengthscale": "single",
    "rank": 1,
    "hadamard_method": True,
}

split_settings: dict = {
    "train_test_splitting": MultipleKFoldSplitter,
    "train_test_splitting_params": {"n_iter": 20, "k": 3},
    "validation_metric": "R^2",
}

# Instantiates the model object
model = MultitaskGaussianProcess(
    prediction_type="regression",
    output_dir=Path.cwd() / "Example_Results",
    n_tasks=5,
    verbose=True,
    hyperparameters_fixed=model_architecture,
    feature_scaler=IdentityScaler
)

# Trains the model and evaluates the performance on the test sets, saves all predictions in the output directory
model.run(
    features=gnn_features.to_numpy(),
    targets=observations.to_numpy(),
    **split_settings
)

