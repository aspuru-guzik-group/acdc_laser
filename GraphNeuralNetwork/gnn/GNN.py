import json
from typing import Union, List, Callable, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
import graph_nets
import sonnet as snt
import tensorflow as tf
from .BatchUtils import get_graph_batch, get_batch_indices
from .EarlyStopping import EarlyStopping
from .Splitters import ECFPSplitter, RandomSplitter


class GNN(snt.Module):
    """
    Graph neural network model using DeepMind's graph_nets architecture.
    """

    def __init__(
            self,
            from_file: Union[bool, Path],
            node_size: Optional[int] = None,
            edge_size: Optional[int] = None,
            global_size: Optional[int] = None,
            n_layers: Optional[int] = None,
            output_dim: Optional[int] = None,
            output_activation: Optional[Callable] = None,
            verbose: bool = False
    ):
        """
        Constructs the GNN by instantiating
            - the encoding function (saved to the 'encode' attribute)
            - the actual GNN layers (saved as the 'gnn' attribute)
            - the prediction layer (saved as the 'prediction_layer' attribute)

        Args:
              node_size: Number of node features.
              edge_size: Number of edge features.
              global_size: Size of the "globals" vector.
              n_layers: Number of GNN layers.
              output_dim: Dimensionality of the output vector (number of targets).
              output_activation: Activation function for the output layer.
              verbose: True if you want to print out the training progress.
        """
        self._job_id = f"graphnet_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        super(GNN, self).__init__(name=self._job_id)

        self._verbose = verbose

        if from_file:
            self._load_trained_model(from_file)

        else:
            self._build_from_scratch(node_size, edge_size, global_size, n_layers, output_dim, output_activation)

        if self._verbose:
            print(f"Initialized a GNN model on {tf.config.list_physical_devices('GPU')[0]}.")

    def _build_from_scratch(
            self,
            node_size: int,
            edge_size: int,
            global_size: int,
            n_layers: int,
            output_dim: int,
            output_activation: Callable
    ):
        """
        Constructs the GNN from scratch by instantiating
            - the encoding function (saved to the 'encode' attribute)
            - the actual GNN layers (saved as the 'gnn' attribute)
            - the prediction layer (saved as the 'prediction_layer' attribute)

        Args:
              node_size: Number of node features.
              edge_size: Number of edge features.
              global_size: Size of the "globals" vector.
              n_layers: Number of GNN layers.
              output_dim: Dimensionality of the output vector (number of targets).
              output_activation: Activation function for the output layer.
        """

        self.encode = graph_nets.modules.GraphIndependent(
            node_model_fn=lambda: snt.Linear(node_size),
            edge_model_fn=lambda: snt.Linear(edge_size)
        )

        self.gnn = snt.Sequential(
            [self._build_gnn_layer(node_size, edge_size, global_size, idx) for idx in range(n_layers)]
        )

        self.prediction_layer = snt.Sequential(
            [snt.Linear(output_dim), output_activation]
        )

        self._model_architecture: dict = {
            "node_size": node_size,
            "edge_size": edge_size,
            "global_size": global_size,
            "n_layers": n_layers,
            "output_dim": output_dim
        }

        self._inference_only = False

    def _build_gnn_layer(
            self,
            node_size: int,
            edge_size: int,
            global_size: int,
            layer_index: int
    ):
        """
        Builds a GNN layer using the GraphNetwork from graph_networks.
        In the input layer (layer_index = 0), the globals are not used for building the GNN.

        Args:
            node_size: Size of the node vector.
            edge_size: Size of the edge vector.
            global_size: Size of the globals vector.
            layer_index: Index of the layer.

        Returns:
            graph_nets.modules.GraphNetwork: GNN layer.
        """
        gnn_layer = graph_nets.modules.GraphNetwork(
            node_model_fn=self._mlp_unit_generator([node_size] * 2),
            edge_model_fn=self._mlp_unit_generator([edge_size] * 2),
            global_model_fn=self._mlp_unit_generator([global_size] * 2),
            node_block_opt={"use_globals": layer_index != 0},
            edge_block_opt={"use_globals": layer_index != 0},
            global_block_opt={"use_globals": layer_index != 0},
            name=f"graphnet_{layer_index}"
        )

        return gnn_layer

    @staticmethod
    def _mlp_unit_generator(
            layers: List[int],
            activation: Callable = tf.nn.relu
    ) -> Callable:
        """
        Returns a function that generates a single MLP unit, composed of multiple fully connected layers and a
        LayerNorm.

        Args:
             layers: List of layer sizes.
             activation: Activation function (default: ReLU)

        Returns:
            Callable: Function that returns a new MLP unit.
        """
        def generate_mlp_unit():
            mlp_unit = snt.Sequential(
                [
                    snt.nets.MLP(layers, activate_final=True, activation=activation),
                    snt.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                ]
            )
            return mlp_unit

        return generate_mlp_unit

    def _load_trained_model(
            self,
            trained_model_dir: Path
    ) -> None:
        """
        Loads a trained model from the specified file directory. Instantiates the "encode", "gnn" and "prediction_layer"
        attributes. Sets the "_inference_only" flag to True (meaning that the model cannot be re-trained).

        Args:
            trained_model_dir: Path to the folder where the model is saved.
                               Must contain the sub-directories "Encoding", "GNN" and "Prediction"
        """
        encoding_layer = tf.saved_model.load(trained_model_dir / "Encoding")
        self.encode = lambda x: encoding_layer.inference(x)

        gnn_layers = tf.saved_model.load(trained_model_dir / "GNN")
        self.gnn = lambda x: gnn_layers.inference(x)

        prediction_layer = tf.saved_model.load(trained_model_dir / "Prediction")
        self.prediction_layer = lambda x: prediction_layer.inference(x)

        self._model_architecture = json.load(open(trained_model_dir / "model_architecture.json", "r"))

        self._inference_only = True

    def embed(
            self,
            features_encoded: Union[graph_nets.graphs.GraphsTuple, List[graph_nets.graphs.GraphsTuple]],
            batch_size: Optional[int] = None,
    ) -> tf.Tensor:
        """
        Embeds a set of graphs (encoded into a GraphsTuple from graph_nets) or batches of sets of graphs (encoded into
        a list of GraphsTuples) it through the GNN layers. If a list of GraphsTuples is passed, the embeddings are
        concatenated along the batch

        Args:
            features_encoded: GraphsTuple object of the graphs that should be embedded, or list of GraphsTuple objects.
            batch_size: Batch size to use for generating the embedding. If None, all graphs are embedded at once.

        Returns:
            tf.Tensor: Embedding of the graphs.
        """
        if isinstance(features_encoded, list):
            return tf.concat([self._embed(graph, batch_size) for graph in features_encoded], axis=0)

        elif isinstance(features_encoded, graph_nets.graphs.GraphsTuple):
            return self._embed(features_encoded, batch_size)

    def _embed(
            self,
            features_encoded: graph_nets.graphs.GraphsTuple,
            batch_size: Optional[int] = None
    ) -> tf.Tensor:
        """
        Embeds a set of graphs (encoded into a GraphsTuple from graph_nets) by passing it through the GNN layers.
        
        Args:
            features_encoded: GraphsTuple object of the graph that should be embedded.
            batch_size: Batch size to use for generating the embedding. If None, all graphs are embedded at once.
        
        Returns:
            tf.Tensor: Embedding of the graphs.
        """
        if batch_size is None:
            return self.gnn(self.encode(features_encoded)).globals
        else:
            batch_indices, residual_indices = get_batch_indices(len(features_encoded), batch_size)
            embeddings = [self.gnn(self.encode(get_graph_batch(features_encoded, batch_idx))).globals for batch_idx in batch_indices]
            embeddings.append(self.gnn(self.encode(get_graph_batch(features_encoded, residual_indices))).globals)
            return tf.concat(embeddings, axis=0)

    def __call__(
            self,
            features_encoded: graph_nets.graphs.GraphsTuple,
            batch_size: Optional[int] = None
    ) -> tf.Tensor:
        """
        Main function to evaluate the GNN and obtain predictions for a graph. Embeds the graph, and passes the embedding
        through the prediction layer.

        Args:
             features_encoded: GraphsTuple object of the graph that should be passed through the network.
            batch_size: Batch size to use for generating the prediction. If None, all data points are run at once.

        Returns:
             tf.Tensor: Output of the predictions.
        """
        if batch_size is None:
            return self.prediction_layer(self.embed(features_encoded))
        else:
            batch_indices, residual_indices = get_batch_indices(len(features_encoded), batch_size)
            predictions = [self.prediction_layer(self.embed(get_graph_batch(features_encoded, batch_idx))) for batch_idx in batch_indices]
            predictions.append(self.prediction_layer(self.embed(get_graph_batch(features_encoded, residual_indices))))
            return tf.concat(predictions, axis=0)

    def train(
            self,
            train_features: graph_nets.graphs.GraphsTuple,
            train_targets: np.ndarray,
            molecule_smiles: List[str],
            validation_size: float = 0.2,
            training_epochs: int = 2000,
            learning_rate: float = 0.01,
            batch_size: int = 256,
            loss_function: Callable = tf.keras.losses.MeanSquaredError(),
            evaluation_metric: Callable = r2_score,
            splitter = ECFPSplitter,
            minimize_evaluation_metric: bool = False,
            early_stopping_patience: int = 100,
            early_stopping_threshold: float = 1E-3,
            sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs batched training of the GNN model.

        Args:
            train_features: GraphsTuple object of all training features.
            train_targets: Numpy ndarray (n_datapoints x n_targets) of training targets.
            molecule_smiles: List of SMILES strings for each datapoint.
            validation_size: Relative size of the validation dataset.
            training_epochs: Number of training epochs.
            learning_rate: Learning rate (default: 0.01).
            batch_size: Size of training batches (default: 256).
            loss_function: Loss function to use for gradient backpropagation (default: MSE).
            evaluation_metric: Function for evaluating the model predictive performance (default: R^2 from sklearn).
            splitter: Function for splitting the dataset into train and validation sets that follows the sklearn API.
                      (Default: ECFPSplitter from deepchem).
            minimize_evaluation_metric: True if the evaluation metric should be minimized.
            early_stopping_patience: Number of iterations without improvement before early stopping is triggered.
            early_stopping_threshold: Minimum improvement in the evaluation function required for early stopping.
            sample_weights: [optional] Numpy array (n_datapoints) to assign weights for each sample.

        Returns:
            np.ndarray: True target values of the validation set.
            np.ndarray: Predicted target values of the validation set.
        """
        if self._inference_only:
            raise NotImplementedError("Only models that are built from scratch can be trained. ")

        # Perform Split in Train and Validation Dataset
        train_idx, validation_idx = list(splitter(molecule_smiles=molecule_smiles, n_splits=1, test_size=validation_size).split(train_targets))[0]
        validation_features = get_graph_batch(train_features, validation_idx)
        train_features = get_graph_batch(train_features, train_idx)
        validation_targets = train_targets[validation_idx, :]
        train_targets = train_targets[train_idx, :]
        validation_weights = sample_weights[validation_idx, :] if sample_weights else None
        train_weights = sample_weights[train_idx, :] if sample_weights else None

        # Setup Optimization
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        early_stopping = EarlyStopping(
            self,
            patience=early_stopping_patience,
            threshold=early_stopping_threshold,
            minimize=minimize_evaluation_metric,
            tmp_dir=Path.cwd() / "tmp" / self._job_id
        )

        if self._verbose:
            progressbar = tqdm(range(training_epochs), desc="Training Epochs")
        else:
            progressbar = range(training_epochs)

        num_iter = 0
        for num_iter in progressbar:

            # Performs batch-wise training using pseudo-random batch indices
            for batch_idx in get_batch_indices(len(train_targets), batch_size)[0]:

                # Gets the features & targets for each batch (requires the get_graph_batch helper to get batches of
                # GraphsTuples
                batch_features: graph_nets.graphs.GraphsTuple = get_graph_batch(train_features, batch_idx)
                batch_targets: tf.Tensor = tf.gather(train_targets, batch_idx)
                batch_weights: tf.Tensor = tf.gather(train_weights, batch_idx) if train_weights is not None else None

                # Calculates the gradients and back-propagates the loss
                with tf.GradientTape() as gradient_tape:
                    predicted_targets = self(batch_features)
                    loss = loss_function(batch_targets, predicted_targets, sample_weight=batch_weights)
                gradients = gradient_tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # Performs batch-wise evaluation of the model on the validation set
            prediction = np.zeros(validation_targets.shape)
            for batch_idx in get_batch_indices(len(validation_targets), batch_size)[0]:
                batch_features: graph_nets.graphs.GraphsTuple = get_graph_batch(validation_features, batch_idx)
                batch_prediction: tf.Tensor = self(batch_features)
                prediction[batch_idx.numpy(), :] = batch_prediction.numpy()

            all_evaluated_indices: np.ndarray = np.where(prediction.any(axis=1))[0]  # account for the fact that get_batch_indices ignores residuals
            metric: float = evaluation_metric(validation_targets[all_evaluated_indices, :], prediction[all_evaluated_indices, :], sample_weight=validation_weights)

            if self._verbose:
                progressbar.set_postfix({"Validation Score": metric})
            elif num_iter % 100 == 0:
                print(f"{num_iter} Iterations Complete: Validation Score: {metric}.")

            if early_stopping.check_convergence(metric):
                break

        print(f"Training finished after {num_iter} epochs. Validation Score: {early_stopping.best_value}.")
        early_stopping.restore_best_model()

        prediction = np.zeros(validation_targets.shape)
        for batch_idx in get_batch_indices(len(validation_targets), batch_size)[0]:
            batch_features: graph_nets.graphs.GraphsTuple = get_graph_batch(validation_features, batch_idx)
            batch_prediction: tf.Tensor = self(batch_features)
            prediction[batch_idx.numpy(), :] = batch_prediction.numpy()

        all_evaluated_indices: np.ndarray = np.where(prediction.any(axis=1))[0]

        return validation_targets[all_evaluated_indices, :], prediction[all_evaluated_indices, :]

    def save_model(
            self,
            save_to: Path,
            num_node_features: int = 41,
            num_edge_features: int = 13
    ) -> None:
        """
        Saves the individual components of the trained GNN model ("encode", "gnn" and "prediction_layer" attributes)
        using tensorflow's saved_model functionality.

        Args:
            save_to: Path to the directory where the model should be saved.
            num_node_features: Number of node features.
            num_edge_features: Number of edge features.
        """
        components: dict = {
            "Encoding": (
                self.encode,
                graph_nets.graphs.GraphsTuple(
                    nodes=tf.TensorSpec(shape=(None, num_node_features), dtype=tf.float32),
                    edges=tf.TensorSpec(shape=(None, num_edge_features), dtype=tf.float32),
                    receivers=tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                    senders=tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                    globals=tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                    n_node=tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                    n_edge=tf.TensorSpec(shape=(None, ), dtype=tf.int32)
                )
            ),
            "GNN": (
                self.gnn,
                graph_nets.graphs.GraphsTuple(
                    nodes=tf.TensorSpec(shape=(None, self._model_architecture["node_size"]), dtype=tf.float32),
                    edges=tf.TensorSpec(shape=(None, self._model_architecture["edge_size"]), dtype=tf.float32),
                    receivers=tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                    senders=tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                    globals=tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                    n_node=tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                    n_edge=tf.TensorSpec(shape=(None, ), dtype=tf.int32)
                )
            ),
            "Prediction": (self.prediction_layer, tf.TensorSpec([None, self._model_architecture["global_size"]]))
        }

        for name, (component, input_signature) in components.items():

            @tf.function(input_signature=[input_signature])
            def inference(x):
                return component(x)

            # Creates a new dummy snt module and saves the inference function to it
            module = snt.Module()
            module.inference = inference
            module.all_variables = component.variables
            tf.saved_model.save(module, save_to / name)

        # Saves the model architecture dictionary
        json.dump(self._model_architecture, open(save_to / "model_architecture.json", "w"))

