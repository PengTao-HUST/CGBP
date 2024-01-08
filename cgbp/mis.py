from itertools import chain
import torch
import torch.nn as nn
import dgl.sparse as dglsp
from itertools import chain
import time

from .model import SAGE_my, GCN_my
from .optim import SGD
from .utils import set_random_seed


def loss_func(bit_str, Q_sp):
    '''
    Calculate the cost of the bit string
    '''
    cost = dglsp.spmm(Q_sp, bit_str) @ bit_str
    return cost


def create_qubo_matrix(g, penalty=2):
    '''
    Create the QUBO matrix
    '''
    nnodes = g.num_nodes()
    idx = torch.Tensor([[i, o] for i, o in zip(
        *g.edges()) if i < o]).type(torch.int64).T
    adj_triu = dglsp.spmatrix(idx, shape=(nnodes, nnodes))
    Q_sp = adj_triu * penalty - dglsp.identity((nnodes, nnodes))
    return Q_sp


def count_violation(bit_str, g):
    '''
    Count the number of edges that are conflicted
    '''
    u, v = g.edges()
    cost = ((bit_str[u] == bit_str[v]) * (bit_str[u] ==
            torch.ones_like(bit_str[u]))).type(torch.int32).sum() / 2
    return cost.item()


def count_mis(bit_str):
    '''
    Count the size of MIS
    '''
    return int(sum(bit_str).item())


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
                     momentum=0.9,
                     optimizer='adam',
                     dropout=0.,
                     prob_threshold=.5,
                     batch_norm=True,
                     weight_decay=0,
                     ):
    '''
    Train the model with the CGBP method

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
        max_mis: The size of MIS
        best_bitstring: The best bitstring of the graph
        nvio: The number of violations
    '''
    st = time.time()
    set_random_seed(seed)
    nnodes = graph_dgl.num_nodes()
    dim_embedding = dim_list[0]
    device = q_torch.device

    # initialize embedding layer
    embed = nn.Embedding(nnodes, dim_embedding).to(device)

    # initialize the GNN model
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

    # Initialize the best_bitstring, best_loss, and loss_list
    best_bitstring = torch.zeros((nnodes,)).to(device)
    best_loss = torch.tensor(float('Inf'))
    loss_list = []
    # Train the model with the CGBP
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
            bitstring = (out.detach().flatten() >= prob_threshold) * 1
            best_bitstring = bitstring

        if epoch % 1000 == 0:
            print('Epoch %d | Loss: %.8f' % (epoch, loss_item))

    # Calculate the size of MIS, the number of violations, and the runtime
    max_mis = count_mis(best_bitstring)
    nvio = count_violation(best_bitstring, graph_dgl)
    et = time.time()
    print('>>> Finish! | MIS: {} | NV: {} | Time: {:.3f}'.format(
        max_mis, nvio, et-st))
    return loss_list, max_mis, best_bitstring, nvio
