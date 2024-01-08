import os
import json
import numpy as np
import torch

from cgbp.utils import generate_zs
from cgbp.gc import train_with_chaos, build_graph_from_file_citation


def main():
    device = torch.device('cuda')
    # Set GNN type
    gnn_type = 'sage'
    # Set maximum number of epochs
    max_epoch = 20000
    # Set list of dimensions
    dim_list = [1000, 1000, 8]

    # Set output directory
    outdir = f'results/pubmed'
    # Create output directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Build graph from file
    g = build_graph_from_file_citation('pubmed').to(device)
    # Get adjacency matrix
    adj_sp = g.adjacency_matrix()

    # Load parameters from json file
    with open("params/pubmed_params.json", "r") as f:
        file = json.load(f)

    # Set the hyperparameters obtained by hyperopt
    params = file[f'pubmed']
    dscale = params['dscale']
    cgbp_epoch = params['cgbp_epoch']
    lr = params['lr']
    optimizer = params['optimizer']
    random_pick = params['random_pick']
    dropout = params['drop']
    embed_only = params['embed_only']
    weight_decay = params['weight_decay']

    # Initialize lists for loss, metrics, and best results
    loss_lists = []
    metric_lists = []
    best_results = []
    # Iterate through seeds
    for seed in range(10):
        # Generate random embeddings
        zs = generate_zs(gnn_type, embed_only, dscale, train_bn=True)
        # Train with CGBP
        loss_list, metric_list, best_result = train_with_chaos(
            g, zs, dim_list, adj_sp, gnn_type=gnn_type, max_epoch=max_epoch, cgbp_epoch=cgbp_epoch, lr=lr, random_pick=random_pick,
            seed=seed, dropout=dropout, dscale=dscale, optimizer=optimizer, weight_decay=weight_decay)
        loss_lists.append(loss_list)
        metric_lists.append(metric_list)
        best_results.append(best_result)
        print('-' * 50)
    # Save loss, metrics, and best results to numpy arrays
    np.save(f'{outdir}/loss.npy', np.array(loss_lists))
    np.save(f'{outdir}_metric.npy', np.array(metric_lists))
    np.save(f'{outdir}_best.npy', np.array(best_results))


if __name__ == '__main__':
    main()
