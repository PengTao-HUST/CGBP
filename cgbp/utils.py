import torch
import numpy as np
import networkx as nx
import dgl


def set_random_seed(value, cudnn=False):
    # Set the random seed
    np.random.seed(value)
    torch.manual_seed(value)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def reg_graph(d, n, seed=1):
    # Create a random regular graph with the given degree and number of nodes
    g = nx.random_regular_graph(d, n, seed=seed)
    # Relabel the nodes to integers
    g = nx.relabel.convert_node_labels_to_integers(g)
    # Convert the graph to a DGL graph
    g = dgl.from_networkx(g)
    # Convert the graph to a bidirected graph
    g = dgl.to_bidirected(g)
    return g


def regular_graph_dim_list(nnodes):
    # Calculate the embedding dimension for the given number of nodes
    dim_embedding = round(np.power(nnodes, 1/3))
    # Calculate the hidden dimension for the given embedding dimension
    hidden_dim = int(dim_embedding/2)
    # Create a list of the embedding, hidden, and output dimensions
    dim_list = [dim_embedding, hidden_dim, 1]
    return dim_list


def generate_zs(gnn_type, embed_only, dscale, train_bn=False):
    # Set z1 and z2 to 0 if only add chaos to the embedding layer
    if not embed_only:
        z1, z2 = 3., 1.
    else:
        z1, z2 = 0., 0.

    # Create a list of 0 for batch normalization
    z_bn = [0.]*4 if train_bn else []
    # Set the z values for the given gnn type
    if gnn_type == 'sage':
        zs = np.array([20., z1, z1, z1, z2, z2, z2] + z_bn) / dscale
    else:
        zs = np.array([20., z1, z1, z2, z2] + z_bn) / dscale

    return zs
