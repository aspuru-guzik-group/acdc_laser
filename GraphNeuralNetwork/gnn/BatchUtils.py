from typing import Tuple
import tensorflow as tf
import graph_nets


def get_batch_indices(
        num_data_points: int,
        batch_size: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Generates pseudo-randomized batches of training data indices that match a given batch size.
    Attention: Ignores the residual data points.

    Args:
          num_data_points: Total number of training data points.
          batch_size: Size of each batch.

    Returns:
        tf.Tensor: Tensor of shape (num_batches, batch_size) containing the indices of the training data points per batch.
        tf.Tensor: Tensor of shape (num_residual_data_points) containing the indices of the residual data points.
    """
    num_batches: int = num_data_points // batch_size
    training_data_size: int = num_batches * batch_size
    indices_shuffled: tf.Tensor = tf.random.shuffle(tf.range(num_data_points))
    random_indices = indices_shuffled[:training_data_size]
    residual_indices = indices_shuffled[training_data_size:]
    return tf.reshape(random_indices, (num_batches, batch_size)), residual_indices


def get_graph_batch(
        full_set: graph_nets.graphs.GraphsTuple,
        batch_indices: tf.Tensor
) -> graph_nets.graphs.GraphsTuple:
    """
    Generates a GraphsTuple object (from graph_nets) that only contains the entries corresponding to a subset of
    indices.

    Args:
        full_set: GraphsTuple object of the full dataset.
        batch_indices: Indices of the entries that should be part of the current batch.

    Returns:
        graph_nets.graphs.GraphsTuple: GraphsTuple object of the batch.
    """
    node_indices = tf.concat([tf.constant([0]), tf.cumsum(full_set.n_node)], axis=0)
    starting_nodes = tf.gather(node_indices, batch_indices)
    ending_nodes = tf.gather(node_indices, batch_indices + 1)
    node_slice = tf.ragged.range(starting_nodes, ending_nodes).values
    batch_nodes = tf.gather(full_set.nodes, node_slice)

    edge_indices = tf.concat([tf.constant([0]), tf.cumsum(full_set.n_edge)], axis=0)
    starting_edges = tf.gather(edge_indices, batch_indices)
    ending_edges = tf.gather(edge_indices, batch_indices + 1)
    edge_slice = tf.ragged.range(starting_edges, ending_edges).values
    batch_edges = tf.gather(full_set.edges, edge_slice) if full_set.edges is not None else None

    num_nodes = tf.gather(full_set.n_node, batch_indices)
    num_edges = tf.gather(full_set.n_edge, batch_indices)

    offsets = tf.repeat(starting_nodes, num_edges)
    senders = tf.gather(full_set.senders, edge_slice) - offsets
    receivers = tf.gather(full_set.receivers, edge_slice) - offsets
    batch_offsets = tf.concat([tf.constant([0]), tf.cumsum(num_nodes)], axis=0)
    batch_senders = senders + tf.repeat(batch_offsets[:-1], num_edges)
    batch_receivers = receivers + tf.repeat(batch_offsets[:-1], num_edges)

    batch_globals = tf.gather(full_set.globals, batch_indices) if full_set.globals is not None else None

    return graph_nets.graphs.GraphsTuple(
        nodes=batch_nodes,
        edges=batch_edges,
        globals=batch_globals,
        senders=batch_senders,
        receivers=batch_receivers,
        n_node=num_nodes,
        n_edge=num_edges
    )
