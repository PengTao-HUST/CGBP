# Chaotic Graph Backpropagation (CGBP)
Graph neural networks (GNNs) with unsupervised learning can provide high-quality approximate solutions tosolve large-scale combinatorial optimization problems (COPs) with efficient time complexity, making them versatile for various applications. However, since this method maps the combinatorial optimization problem to the training process of a graph neural network, and the current mainstream backpropagation-based training algorithms are prone to fall into local minima, the optimization performance is still inferior to the current state-of-the-art (SOTA) COP methods. To address this issue, inspired by possibly chaotic dynamics of real brain learning, we introduce a chaotic training algorithm, i.e., chaotic graph backpropagation (CGBP), which introduces a local loss function in GNN that makes the training process not only chaotic but also highly efficient. Different from existing methods, we show that the global ergodicity and pseudo-randomness of such chaotic dynamics enable CGBP to learn GNNs effectively and globally, thus solving the COP efficiently. We have applied CGBP to solve various COPs, such as the maximum independent set, maximum cut, and graph coloring. Results on several large-scale benchmark datasets showcase that CGBP can compete with or outperform SOTA methods. In addition to solving large-scale COPs, CGBP as a universal learning algorithm for GNNs can be easily integrated into any existing method for improving the performance.

## Requirements
- torch >= 2.0
- dgl >= 1.0
- networkx >= 3.0
- hyperopt >= 0.2 (opt)

## Usage
Simply run
```
python xxx.py
```
in this directory and the results will be saved in the [results](https://github.com/PengTao-HUST/CGBP/tree/master/results) folder. Please note that while it is possible to set a random number seed in the program, the results still exhibit inherent randomness. The results you obtain may differ from those in the [results](https://github.com/PengTao-HUST/CGBP/tree/master/results) folder, but typically, there should not be significant differences.

## Highlighted results
### 1. Max cut
Running [plot_mc.ipynb](https://github.com/PengTao-HUST/CGBP/tree/master/plot_mc.ipynb) you will get the following two figures comparing the performance (approximate ratio of theoretical values and running time) of CGBP and the original PI-GNN and greedy algorithms. 

![figure](https://github.com/PengTao-HUST/CGBP/blob/master/figs/mcut_ratio.png?raw=true)

![figure](https://github.com/PengTao-HUST/CGBP/blob/master/figs/mcut_time.png?raw=true)

### 2. Max independent set
Running [plot_mis.ipynb](https://github.com/PengTao-HUST/CGBP/tree/master/resultsplot_mis.ipynb) you will get the following two figures comparing the performance (approximate ratio of theoretical values and running time) of CGBP and the original PI-GNN and greedy algorithms.

![figure](https://github.com/PengTao-HUST/CGBP/blob/master/figs/mis_ratio.png?raw=true)

![figure](https://github.com/PengTao-HUST/CGBP/blob/master/figs/mis_time.png?raw=true)

### 3. Graph coloring
Running [plot_pubmed.ipynb](https://github.com/PengTao-HUST/CGBP/tree/master/plot_pubmed.ipynb) you will get the following optimal coloring result of the Pubmed graph.

![figure](https://github.com/PengTao-HUST/CGBP/blob/master/figs/pubmed_coloring.png?raw=true)

## License
MIT License