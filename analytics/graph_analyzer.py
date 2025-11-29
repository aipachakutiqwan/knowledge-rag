import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, degree, to_networkx
import networkx as nx

def extract_graphs(subgraphs):
    subgraph_list = []
    for i in subgraphs:
        df = pd.read_csv(f"expla_graphs/edges/{i}.csv")   # contains src, edge_attr, dst
        edge_index = torch.tensor([df['src'].tolist(), df['dst'].tolist()], dtype=torch.long)
        edge_attr = df['edge_attr'].tolist()
        node_df = pd.read_csv(f"expla_graphs/nodes/{i}.csv")   # contains src, edge_attr, dst
        data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes = len(node_df['node_id']))
        subgraph_list.append(data)
    return subgraph_list

def get_stats_distributions(graphs):
    densities = []
    centralities = []
    diameters = []
    sizes = []
    for graph in graphs:
        densities.append((2 * graph.edge_index.shape[1]) / (graph.num_nodes * (graph.num_nodes - 1)))
        nx_graph = to_networkx(graph).to_undirected()
        closeness_centralities = nx.closeness_centrality(nx_graph)
        centralities.append(sum(closeness_centralities.values()) / (graph.num_nodes) * (graph.num_nodes - 1))
        diameters.append(nx.diameter(nx_graph))
        sizes.append(graph.num_nodes)

    return densities, centralities, diameters, sizes

def plot_densities(densities):
    xticks = np.arange(0, 1.1, 0.1)
    plt.hist(densities, bins=xticks, edgecolor='k')
    plt.xlabel('Density (proportion of possible edges that exist)')
    plt.xticks(xticks)
    plt.ylabel('Number of subgraphs')
    plt.title('Distribution of Densities on Expla Graphs')
    plt.savefig('expla_densities.png')
    plt.clf()

def plot_centralities(centralities):
    xticks = np.arange(2, 3.75, 0.25)
    plt.hist(centralities, bins=xticks, edgecolor='k')
    plt.xlabel('Global Efficiency')
    plt.xticks(xticks)
    plt.ylabel('Number of subgraphs')
    plt.title('Distribution of Global Efficiencies in Expla Graphs')
    plt.savefig('expla_efficiencies.png')
    plt.clf()

def plot_diameters(diameters):
    xticks = np.arange(8)
    plt.hist(diameters, bins=xticks, edgecolor='k')
    plt.xlabel('Diameters')
    plt.xticks(xticks)
    plt.ylabel('Number of subgraphs')
    plt.title('Distribution of Diameters in Expla Graphs')
    plt.savefig('expla_diameters.png')
    plt.clf()

def plot_subgraph_sizes(sizes):
    xticks = np.arange(8)
    plt.hist(sizes, bins=xticks, edgecolor='k')
    plt.xlabel('Diameters')
    plt.xticks(xticks)
    plt.ylabel('Number of subgraphs')
    plt.title('Distribution of Subgraph Sizes in Expla Graphs')
    plt.savefig('expla_sizes.png')
    plt.clf()

graphs = extract_graphs(np.arange(len(os.listdir("expla_graphs/edges"))))
densities, centralities, diameters, sizes = get_stats_distributions(graphs)

plot_densities(densities)
plot_centralities(centralities)
plot_diameters(diameters)
plot_subgraph_sizes(sizes)

