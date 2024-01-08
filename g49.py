import time
import os
import json
import numpy as np
import torch

from cgbp.utils import generate_zs
from cgbp.mc import read_gset_file, train_with_chaos


def main():
    device = torch.device('cuda')
    # Set the GNN type
    gnn_type = 'sage'
    # Set the maximum number of epochs
    max_epoch = 20000
    # Set the dimensions of the model
    dim_list = [100, 100, 1]

    # Read the graph and the adjacency matrix
    g, Q_mat = read_gset_file(name=f'./data/G49.txt')
    # Move the graph and the adjacency matrix to the device
    g = g.to(device)
    Q_mat = Q_mat.to(device)

    # Set the output directory
    outdir = f'results/g49'
    # Create the output directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Open the parameters file
    with open("params/g49_params.json", "r") as f:
        file = json.load(f)

    # Set the hyperparameters obtained by hyperopt
    params = file[f'g49']
    dscale = params['dscale']
    cgbp_epoch = params['cgbp_epoch']
    lr = params['lr']
    optimizer = params['optimizer']
    random_pick = params['random_pick']
    dropout = params['drop']
    embed_only = params['embed_only']
    weight_decay = params['weight_decay']

    # Create empty lists to store the time and maximum cut values
    time_lists = []
    mc_lists = []
    # Iterate through the seeds
    for seed in range(10):
        # Generate the chaotic intensities
        zs = generate_zs(gnn_type, embed_only, dscale)
        stime = time.time()
        # Train the model with CGBP
        loss_list, max_cut, best_bitstring = train_with_chaos(
            g, Q_mat, zs, dim_list, gnn_type=gnn_type, max_epoch=max_epoch, cgbp_epoch=cgbp_epoch, 
            lr=lr, random_pick=random_pick, seed=seed, dropout=dropout, dscale=dscale, 
            optimizer=optimizer, weight_decay=weight_decay)
        etime = time.time()
        rtime = etime - stime
        time_lists.append(rtime)
        mc_lists.append(max_cut)

        print('-' * 50)
    # Save the time and maximum cut values to the output directory
    np.save(f'{outdir}/time.npy', np.array(time_lists))
    np.save(f'{outdir}/mcut.npy', np.array(mc_lists))


if __name__ == '__main__':
    main()