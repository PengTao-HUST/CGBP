import time
import os
import json
import numpy as np
import torch

from cgbp.utils import reg_graph, generate_zs, regular_graph_dim_list
from cgbp.mis import create_qubo_matrix, train_with_chaos


def main():
    device = torch.device('cuda')
    # Set the degree of the graph
    d = 3
    # Set the type of GNN
    gnn_type = 'gcn'
    # Set the maximum number of epochs
    max_epoch = 20000

    # Open the parameters file
    with open("params/mis_d3_params.json", "r") as f:
        file = json.load(f)

    # Iterate through the number of nodes
    for r in range(2, 7):
        # Set the number of nodes
        n = 10 ** r
        # Create the graph
        g = reg_graph(d, n).to(device)
        # Create the QUBO matrix
        Q_mat = create_qubo_matrix(g).to(device) / g.num_nodes()
        # Create the dimension list
        dim_list = regular_graph_dim_list(g.num_nodes())

        # Set the output directory
        outdir = f'results/mis/n{n}_d{d}_{gnn_type}'
        # Create the output directory if it does not exist
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Set the hyperparameters obtained by hyperopt
        params = file[f'd3_n{n}']
        dscale = params['dscale']
        cgbp_epoch = params['cgbp_epoch']
        lr = params['lr']
        optimizer = params['optimizer']
        random_pick = params['random_pick']
        dropout = params['drop']
        embed_only = params['embed_only']
        weight_decay = params['weight_decay']

        # Create empty lists for time and mis
        time_lists = []
        mis_lists = []
        # Iterate through the number of seeds
        for seed in range(10):
            # Generate zs (chaotic intensities)
            zs = generate_zs(gnn_type, embed_only, dscale)
            stime = time.time()
            # Train with CGBP
            loss_list, max_mis, best_bitstring, nvio = train_with_chaos(
                g, Q_mat, zs, dim_list, gnn_type=gnn_type, max_epoch=max_epoch, cgbp_epoch=cgbp_epoch,
                lr=lr, random_pick=random_pick, seed=seed, dropout=dropout, dscale=dscale,
                optimizer=optimizer, weight_decay=weight_decay)
            etime = time.time()
            rtime = etime - stime
            time_lists.append(rtime)
            mis_lists.append(max_mis)

            print('-' * 50)

        # Save the results
        np.save(f'{outdir}/time.npy', np.array(time_lists))
        np.save(f'{outdir}/mis.npy', np.array(mis_lists))


if __name__ == '__main__':
    main()
