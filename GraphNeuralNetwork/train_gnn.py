import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from gnn.MoleculeEncoder import MolFeaturizer
from gnn.GNN import GNN


# load the training dataset and transform the targets using a MinMaxScaler
dft_dataset = np.load(
    "data/full_dataset_filtered_r_0.98_descriptors_selected.npz",
    allow_pickle=True
)

smiles = list(dft_dataset["smiles"])
targets = dft_dataset["descriptors"]
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
