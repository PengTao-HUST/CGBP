import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
import dgl
import time

from .model import SAGE_my, GCN_my
from .optim import SGD
from .utils import set_random_seed


def build_graph_from_file_dimacs(fpath):
    # open the file and read the content
    with open(fpath, 'r') as f:
        content = f.read().strip().split('\n')

    # initialize the list of x and y coordinates
    xs = []
    ys = []
    # loop through each line in the content
    for line in content:
        # if the line starts with 'e', get the x and y coordinates and append them to the list
        if line.startswith('e'):
            x, y = line.split()[1:]
            xs.append(int(x) - 1)
            ys.append(int(y) - 1)
        # if the line starts with 'p', get the number of nodes and edges
        elif line.startswith('p'):
            nnode, nedge = line.split()[2:]
            nnode = int(nnode)
            nedge = int(nedge)
        # otherwise, do nothing
        else:
            pass

    # create a graph using the x and y coordinates
    g = dgl.graph((xs, ys))
    # convert the graph to a bidirected graph
    g = dgl.to_bidirected(g)

    # print the number of nodes and edges
    print('-'*50)
    print('  Number of nodes:', g.num_nodes())
    print('  Number of edges:', g.num_edges())

    # check if the graph contains self-loops
    if (np.array(xs) == np.array(ys)).astype(int).sum() != 0:
        print(f'  warning: containing self-loops.')
    # check if the graph contains 0-degree nodes
    if 0 in g.in_degrees():
        print(f'  warning: containing 0-degree nodes.')
    print('-'*50)
    return g


def build_graph_from_file_citation(name):
    # check if the name is 'cora', 'citeseer' or 'pubmed'
    if name == 'cora':
        # if the name is 'cora', get the graph from the CoraGraphDataset
        g = dgl.data.CoraGraphDataset(raw_dir='./data/')[0]
    elif name == 'citeseer':
        # if the name is 'citeseer', get the graph from the CiteseerGraphDataset
        g = dgl.data.CiteseerGraphDataset(raw_dir='./data/')[0]
    elif name == 'pubmed':
        # if the name is 'pubmed', get the graph from the PubmedGraphDataset
        g = dgl.data.PubmedGraphDataset(raw_dir='data/')[0]
    else:
        # if the name is not one of the three, raise a ValueError
        raise ValueError(f'Unknown dataset name {name}.')
    # remove the self-loops from the graph
    g = dgl.remove_self_loop(g)

    # print the number of nodes and edges
    print('-'*50)
    print('  Number of nodes:', g.num_nodes())
    print('  Number of edges:', g.num_edges())
    # get the x and y coordinates of the edges
    xs, ys = g.edges()
    # check if the graph contains self-loops
    if (np.array(xs) == np.array(ys)).astype(int).sum() != 0:
        print(f'  warning: containing self-loops.')
    # check if the graph contains 0-degree nodes
    if 0 in g.in_degrees():
        print(f'  warning: containing 0-degree nodes.')
    print('-'*50)
    # return the graph
    return g


def loss_func_mod(probs, adj_tensor):
    # calculate the loss
    loss_ = torch.mul(adj_tensor, (probs @ probs.T)).sum() / 2
    # return the loss
    return loss_


def loss_func_color_hard(coloring, adj_sp):
    # get the indices of the edges
    u, v = adj_sp.indices()
    # calculate the cost
    cost = (coloring[u] == coloring[v]).type(torch.int32).sum() / 2
    # return the cost
    return int(cost.item())


def train_with_chaos(dgl_graph,
                     zs,
                     dim_list,
                     adj_sp,
                     gnn_type='gcn',
                     max_epoch=20000,
                     cgbp_epoch=10000,
                     lr=1e-4,
                     beta=0.999,
                     random_pick=False,
                     batch_norm=True,
                     spe_idx=0,
                     dscale=1.,
                     cgbp_momentum=0.,
                     momentum=0.,
                     optimizer='adam',
                     dropout=0.0,
                     weight_decay=0.01,
                     print_prog=True,
                     print_itv=1000,
                     seed=1
                     ):
    '''
    Train with the CGBP method

    Args:
        graph_dgl: The input dgl graph
        zs: The initial value of the chaotic intensities
        dim_list: The list of the dimensions of the model
        adj_sp: The sparse adjacent matrix
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
        print_prog: The flag for print the traning process
        print_itv: The interval of print

    Returns:
        loss_list: The list of the training loss
        metric_list: The list of the number of violations
        best_result: The best coloring of the graph
    '''
    # set random seed
    st = time.time()
    set_random_seed(seed)
    device = dgl_graph.device
    adj = adj_sp.to_dense()
    # initialize embedding layer
    embed = nn.Embedding(dgl_graph.num_nodes(), dim_list[0]).to(device)

    # initialize model
    if gnn_type == 'sage':
        model = SAGE_my(dim_list, dropout=dropout, dscale=dscale, batch_norm=batch_norm,
                        random_pick=random_pick, spe_idx=spe_idx, last_sigmoid=False).to(device)
    else:
        model = GCN_my(dim_list, dropout=dropout, dscale=dscale, batch_norm=batch_norm,
                       random_pick=random_pick, spe_idx=spe_idx, last_sigmoid=False).to(device)
    # initialize CGBP optimizer
    optimizer = SGD(chain(embed.parameters(), model.parameters()), lr=lr, momentum=cgbp_momentum,
                    weight_decay=weight_decay)
    inp = embed.weight

    # initialize optimal solution
    best_cost = torch.tensor(float('Inf'))
    best_epoch = 0
    best_result = None
    loss_list = []
    metric_list = []
    # start training
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
        out, hid = model(dgl_graph, inp)
        probs = F.softmax(out, dim=1)
        loss = loss_func_mod(probs, adj)
        loss_item = loss.item()
        loss_list.append(loss_item)

        # calculate the violations
        result = torch.argmax(probs, dim=1)
        cost_hard = loss_func_color_hard(result, adj_sp)
        metric_list.append(cost_hard)

        # save the best result
        if cost_hard < best_cost:
            best_cost = cost_hard
            best_result = result
            best_epoch = epoch

        # early stop
        # if cost_hard == 0 or cost_hard == dgl_graph.num_edges() // 2:
        #     print(f'>>> Early stop traning at epoch {epoch} ...')
        #     break

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

        # print the progress
        if print_prog and epoch % print_itv == 0:
            print('Epoch %d | Soft Loss: %.5f | Hard Cost: %d' %
                  (epoch, loss.item(), cost_hard))

    et = time.time()
    if print_prog:
        print('>>> Finish! | Best: {} | Epoch: {} | Time: {:.3f} sec'.format(
            int(best_cost), best_epoch, et-st))

    return loss_list, metric_list, best_result
