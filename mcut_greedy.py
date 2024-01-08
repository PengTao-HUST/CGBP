import numpy as np
import time
import torch
import os
import dgl.sparse as dglsp

from cgbp.utils import reg_graph, set_random_seed
from cgbp.mc import create_qubo_matrix, count_cuts


def loss_func(bit_str, Q_sp):
    # Calculate the cost of the bit string by multiplying the QUBO matrix with the bit string
    cost = dglsp.spmm(Q_sp, bit_str) @ bit_str
    return cost


def count_cuts(bit_str, Q_sp):
    # Return the number of cuts in the graph by taking the negative of the loss function
    return int(-loss_func(bit_str, Q_sp).item())


def run_experiment(n, Q_mat, seed, device, mode):
    # Set the maximum number of epochs to 10 times the number of nodes
    max_epoch = 10 * n

    stime = time.time()
    set_random_seed(seed)
    # Generate a random bit string of length n
    bit_str = torch.randint(2, size=(n,)).float().to(device)
    # Calculate the maximum cut
    maxcut = count_cuts(bit_str, Q_mat)

    print(f'Seed {seed} | Epoch 0/{max_epoch} | MC {maxcut}')
    # Create an empty list to store the cuts
    cut_list = []
    # Set the best cut to 0
    best_cut = 0
    # Start the epoch loop
    for epoch in range(max_epoch):
        # Create an empty list to store the number of cuts after flipping each bit
        ncut_flip_list = []
        # If the mode is fast, randomly select 100 bits to flip
        if mode == 'fast':
            sele_list = np.random.choice(range(n), 100, replace=False)
        # Otherwise, randomly permute the bits
        else:
            sele_list = np.random.permutation(range(n))
        # Start the flip loop
        for i in sele_list:
            # Create a copy of the bit string
            bit_str_flip = bit_str.clone()
            # Flip the bit at index i
            bit_str_flip[i] = 1 - bit_str_flip[i]
            # Calculate the number of cuts after flipping the bit
            ncut_flip = count_cuts(bit_str_flip, Q_mat)
            # Append the number of cuts to the list
            ncut_flip_list.append(ncut_flip)
        # Find the index of the bit to flip
        flip_idx = sele_list[np.argmax(ncut_flip_list)]
        # Set the maximum cut to the maximum number of cuts after flipping the bit
        maxcut = np.max(ncut_flip_list)
        # Append the maximum cut to the list
        cut_list.append(maxcut)
        # Flip the bit at the index of the bit to flip
        bit_str[flip_idx] = 1 - bit_str[flip_idx]
        # If the maximum cut is greater than the best cut, set the best string to the current bit string
        if maxcut > best_cut:
            best_str = bit_str

        if (epoch+1) % n == 0:
            print(f'Seed {seed} | Epoch {epoch+1}/{max_epoch} | MC {maxcut}')

    rtime = time.time() - stime
    best_cut = np.max(cut_list)
    return rtime, best_cut, best_str.cpu().numpy()


def main():
    for r in range(2, 6):
        # Set n to 10^r
        n = 10 ** r 
        d = 3 # degree
        mode = 'fast'  # fast or full

        # Create the output directory
        outdir = f'results/mc_greedy/n{n}_d{d}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Set the device to cuda if available, otherwise use cpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create a regular graph with the given number of nodes and degree
        g = reg_graph(d, n).to(device)
        # Create the QUBO matrix
        Q_mat = create_qubo_matrix(g)

        # Create empty lists to store the best cut, best string, and runtime
        best_cut_list = []
        best_str_list = []
        time_list = []
        # Set the number of times to repeat the experiment
        n_repeat = 10
        # Start the loop over the number of times to repeat the experiment
        for seed in range(n_repeat):
            # Run the experiment and get the runtime, best cut, and best string
            rtime, best_cut, best_str = run_experiment(
                n, Q_mat, seed, device, mode)
            print('>>> Finish! | Maxcut: {} | Time: {:.3f}'.format(best_cut, rtime))
            print('-' * 50)
            time_list.append(rtime)
            best_cut_list.append(best_cut)
            best_str_list.append(best_str)

        # Save the runtime and best cut to the output directory
        np.save(f'{outdir}/time.npy', time_list)
        np.save(f'{outdir}/mcut.npy', best_cut_list)


if __name__ == '__main__':
    main()
