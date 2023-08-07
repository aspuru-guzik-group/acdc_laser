import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from gnn.MoleculeEncoder import MolFeaturizer
from gnn.GNN import GNN


# load the training dataset and transform the targets using a MinMaxScaler
dft_dataset = pd.read_csv("Data/computational_seed_dataset/tddft_dataset_filtered_selected.csv", index_col=False)
smiles = dft_dataset.loc[:, "smiles"].tolist()
targets = dft_dataset.iloc[:, 1:].to_numpy(dtype=np.float64)
target_scaler = MinMaxScaler()
targets_scaled = target_scaler.fit_transform(targets)

# load the encoder settings and encode the molecules
encoder_config = json.load(open("used_settings/atom_edge_features.json"))
featurizer = MolFeaturizer(feature_config=encoder_config, n_jobs=12, verbose=False)
graph_features = featurizer.encode_molecules(smiles)

# load the model architecture and instantiate the model
model_architecture = json.load(open("used_settings/model_architecture.json"))

gnn_model = GNN(
    **model_architecture,
    output_dim=targets_scaled.shape[1],
    output_activation=tf.identity,
    from_file=False,
    verbose=False
)

# train the model and save the predictions on the held-out validation set
validation_true, validation_predicted = gnn_model.train(
    train_features=graph_features,
    train_targets=targets_scaled,
    molecule_smiles=smiles,
    training_epochs=5000,
    learning_rate=1E-3,
    batch_size=128,
    early_stopping_patience=200,
)

validation_true = target_scaler.inverse_transform(validation_true)
validation_predicted = target_scaler.inverse_transform(validation_predicted)

np.savez_compressed(
    "data/validation_predictions.npz",
    true_values=validation_true,
    predicted_values=validation_predicted
)

# save the model
gnn_model.save_model(Path("model/trained_model"), num_node_features=41, num_edge_features=13)

