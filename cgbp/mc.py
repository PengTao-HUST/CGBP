import dgl
import torch
import torch.nn as nn
import dgl.sparse as dglsp
import pandas as pd
from itertools import chain
import time

from .model import SAGE_my, GCN_my
from .optim import SGD
from .utils import set_random_seed


# Define a function to calculate the loss
def loss_func(bit_str, Q_sp):
    # Calculate the cost by multiplying the adjacency matrix with the bit string
    cost = dglsp.spmm(Q_sp, bit_str) @ bit_str
    return cost


# Define a function to create the QUBO matrix
def create_qubo_matrix(g):
    # Create a diagonal matrix of the in-degrees of the graph
    indegree_diag = dglsp.diag(g.in_degrees()).float()
    # Create the adjacency matrix of the graph minus the diagonal matrix
    Q_sp = g.adjacency_matrix() - indegree_diag
    return Q_sp


# Define a function to count the number of cuts
def count_cuts(bit_str, g):
    # Get the source and destination nodes of the graph
    u, v = g.edges()
    # Calculate the number of cuts by comparing the source and destination nodes of the graph
    cost = (bit_str[u] != bit_str[v]).sum() // 2
    return cost.item()


# Define a function to read a gset file
def read_gset_file(name):
    with open(name, 'r') as file:
        first_line = file.readline()
        nnode, nedge, _ = first_line.split(' ')
        nnode = int(nnode)
        nedge = int(nedge)

    # Read the file using pandas
    file_ = pd.read_csv(name, sep=' ', skiprows=1, header=None)
    # Get the source nodes
    src = torch.tensor(file_[0].values) - 1
    # Get the destination nodes
    dst = torch.tensor(file_[1].values) - 1
    # Create a graph using the source and destination nodes
    g = dgl.graph((src, dst))
    # Create a bidirectional graph
    g = dgl.to_bidirected(g)

    # Get the weights of the edges
    weights = torch.tensor(file_[2].values).float()

    w_adj = torch.zeros((nnode, nnode))
    w_adj[src, dst] = weights
    w_adj = w_adj + w_adj.T

    # Create the QUBO matrix by subtracting the diagonal matrix from the adjacency matrix
    Q_sp = w_adj - torch.diag(w_adj.sum(0))
    # Create the QUBO matrix from the torch sparse matrix
    Q_sp = dglsp.from_torch_sparse(Q_sp.to_sparse())
    return g, Q_sp


def train_with_chaos(graph_dgl,
                     q_torch,
                     zs,
                     dim_list,
                     gnn_type='gcn',
                     max_epoch=20000,
                     cgbp_epoch=10000,
                     lr=1e-4,
                     beta=0.999,
                     seed=1,
                     random_pick=False,
                     spe_idx=0,
                     dscale=1.,
                     cgbp_momentum=0.,
                     momentum=0.,
                     optimizer='adam',
                     dropout=0.,
                     prob_threshold=.5,
                     batch_norm=True,
                     weight_decay=0,
                     ):
    '''
    Train with the CGBP method

    Args:
        graph_dgl: The input dgl graph
        q_torch: The QUBO matrix
        zs: The initial value of the chaotic intensities
        dim_list: The list of the dimensions of the model
        gnn_type: The type of GNN, can be 'gcn' or 'sage'
        max_epoch: The maximum number of epochs
        cgbp_epoch: The number of epochs for CGBP
        lr: The learning rate
        beta: The parameter for the annealing of zs
        seed: The random seed
        random_pick: The flag for the random pick of nodes
        spe_idx: The index of the specific node
        dscale: The steepness factor for the sigmoid function
        cgbp_momentum: The momentum of CGBP
        momentum: The momentum of BP
        optimizer: The optimizer, can be 'adam' or 'sgd'
        dropout: The dropout rate
        prob_threshold: The probability threshold
        batch_norm: The flag for batch normalization
        weight_decay: The weight decay

    Returns:
        loss_list: The list of the training loss
        max_cut: The maximum cut
        best_bitstring: The best bitstring of the graph
    '''
    st = time.time()
    # set random seed
    set_random_seed(seed)
    nnodes = graph_dgl.num_nodes()
    dim_embedding = dim_list[0]
    device = q_torch.device
    # initialize embedding layer
    embed = nn.Embedding(nnodes, dim_embedding).to(device)

    # initialize model
    if gnn_type == 'sage':
        model = SAGE_my(dim_list, dropout=dropout, dscale=dscale, batch_norm=batch_norm,
                        random_pick=random_pick, spe_idx=spe_idx).to(device)
    else:
        model = GCN_my(dim_list, dropout=dropout, dscale=dscale, batch_norm=batch_norm,
                       random_pick=random_pick, spe_idx=spe_idx).to(device)
        
    # initialize CGBP optimizer
    optimizer = SGD(chain(embed.parameters(), model.parameters()), lr=lr, momentum=cgbp_momentum,
                    weight_decay=weight_decay)
    inp = embed.weight

    # initialize optimal solution
    best_bitstring = torch.zeros((nnodes,)).to(device)
    best_loss = torch.tensor(float('Inf'))
    loss_list = []
    for epoch in range(max_epoch):
        # switch to BP scheme
        if epoch == cgbp_epoch:
            if optimizer == 'adam':
                optimizer = torch.optim.AdamW(chain(embed.parameters(), model.parameters()), lr=lr,
                                              weight_decay=weight_decay)
            else:
                optimizer = torch.optim.SGD(chain(embed.parameters(), model.parameters()), lr=lr,
                                            momentum=momentum, weight_decay=weight_decay)
        # forward pass
        out, hid = model(graph_dgl, inp)
        loss = loss_func(out[:, 0], q_torch)
        loss_item = loss.item()
        loss_list.append(loss_item)

        # calculate the gradient
        optimizer.zero_grad()
        loss.backward()

        # update the parameters
        if epoch < cgbp_epoch:
            optimizer.step(zs, hid)
        else:
            optimizer.step()
            
        # update the zs
        zs *= beta

        # save the best result
        if loss_item < best_loss:
            best_loss = loss_item
            best_bitstring = (out.detach().flatten() >= prob_threshold) * 1

        if epoch % 1000 == 0:
            print('Epoch %d | Loss: %.5f' % (epoch, loss_item))

    max_cut = count_cuts(best_bitstring, graph_dgl)
    et = time.time()
    print('>>> Finish! | Maxcut: {} | Time: {:.3f}'.format(max_cut, et-st))
    return loss_list, max_cut, best_bitstring
