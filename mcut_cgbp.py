import time
import os
import json
import numpy as np
import torch

from cgbp.utils import reg_graph, generate_zs, regular_graph_dim_list
from cgbp.mc import create_qubo_matrix, train_with_chaos


def main():
    device = torch.device('cuda')
    # Set degree to 3
    d = 3
    # Set gnn type
    gnn_type = 'gcn'
    # Set max epoch
    max_epoch = 20000

    # Open params.json file
    with open("params/mc_d3_params.json", "r") as f:
        file = json.load(f)

    for r in range(2, 7):
        # Set n to 10^r
        n = 10 ** r
        # Create regular graph with n nodes
        g = reg_graph(d, n).to(device)
        # Create qubo matrix
        Q_mat = create_qubo_matrix(g)
        # Get dimension list
        dim_list = regular_graph_dim_list(g.num_nodes())

        # Set output directory
        outdir = f'results/mc/n{n}_d{d}_{gnn_type}'
        # Create directory if it does not exist
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

        # Create empty lists for time and mc
        time_lists = []
        mc_lists = []
        # Iterate through 10 seeds
        for seed in range(10):
            # Generate zs (chaotic intensities)
            zs = generate_zs(gnn_type, embed_only, dscale)
            stime = time.time()
            # Train with CGBP
            loss_list, max_cut, best_bitstring = train_with_chaos(
                g, Q_mat, zs, dim_list, gnn_type=gnn_type, max_epoch=max_epoch, cgbp_epoch=cgbp_epoch, 
                lr=lr, random_pick=random_pick, seed=seed, dropout=dropout, dscale=dscale, 
                optimizer=optimizer, weight_decay=weight_decay)
            etime = time.time()
            rtime = etime - stime
            time_lists.append(rtime)
            mc_lists.append(max_cut)

            print('-' * 50)
        # Save time_lists and mc_lists to numpy files
        np.save(f'{outdir}/time.npy', np.array(time_lists))
        np.save(f'{outdir}/mcut.npy', np.array(mc_lists))


if __name__ == '__main__':
    main()