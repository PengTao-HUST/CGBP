import networkx as nx
import numpy as np
import time
import logging
import os


# Define a function to solve the Maximum Independent Set problem using the greedy algorithm
def greedy_max_independent_set(graph, max_degree=False):
    # Initialize an empty set to store the Independent Set
    independent_set = set()
    graph = graph.copy()
    # Iterate until the graph is empty
    while graph:
        # Choose a random node from the current graph
        if max_degree:
            # If max_degree is set to True, choose the node with the highest degree
            node = max(graph.degree, key=lambda item: item[1])[0]
        else:
            # Otherwise, choose a node randomly
            node = np.random.choice(list(graph.nodes()))

        # Add the chosen node to the independent set
        independent_set.add(node)

        # Remove the chosen node and its neighbors from the graph
        neighbors_to_remove = set(graph.neighbors(node)) | {node}
        graph.remove_nodes_from(neighbors_to_remove)

    return independent_set


def main():
    outdir = f'results/mis_greedy/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    logging.basicConfig(filename='mis_greedy.out', level=logging.DEBUG)
    print = logging.debug

    for nnode in [1e2, 1e3, 1e4, 1e5, 1e6]:
        print(f'Creat 3-regular graph with {int(nnode)} nodes.')
        # Create a 3-regular graph with the given number of nodes
        G = nx.random_regular_graph(3, int(nnode), seed=1)
        # Solve the Maximum Independent Set problem using the greedy algorithm
        for seed in range(10):
            np.random.seed(seed)
            st = time.time()
            max_independent_set = greedy_max_independent_set(G)
            et = time.time()
            print(f'seed {seed} | MIS {len(max_independent_set)} | Time {et-st:.4f}')
        print('-' * 50)


if __name__ == "__main__":
    main()
