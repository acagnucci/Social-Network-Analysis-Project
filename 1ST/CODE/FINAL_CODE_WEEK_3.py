#-----------------------------------------------------------WEEK 3
#QUERY 3:
#Depending on what seems more relevant in your graph of the following local notions
#- Closeness centrality
#1) Provide a code computing the given centrality using basic functions of networkx (you are not allowed to use directly nx."what you want").
#2) Discuss why you picked this measure and who is the most central in your network based on your choice.
#3) Provide the cumulative distribution for this centrality and give a graphical representation of your graph (log-log, log or normal representation as you think it is more relevant).
#-----------------------------------------------------------

import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#IMPORTING DATABASE
edges_filename = "./edges.csv"
nodes_filename = "./nodes.csv"

df_edges = pd.read_csv(edges_filename)
df_nodes = pd.read_csv(nodes_filename)


# Create the graph using the edges CSV, same as WEEK 1
G = nx.from_pandas_edgelist(df_edges, '# source', ' target')  # Adjusted column name

# Closeness Centrality
def compute_closeness_centrality(graph):
    closeness_centrality = {}
    for node in graph.nodes():
        total_distance = sum(nx.shortest_path_length(graph, node, target) for target in graph.nodes() if target != node)
        num_nodes = len(graph.nodes()) - 1  # Excluding the node itself
        closeness_centrality[node] = num_nodes / total_distance if total_distance != 0 else 0
    return closeness_centrality

closeness_centrality_values = compute_closeness_centrality(G)
most_central_node_closeness = max(closeness_centrality_values, key=closeness_centrality_values.get)

# Cumulative Distribution Visualization
import numpy as np
import matplotlib.pyplot as plt

def plot_cumulative_distribution(values, title):
    sorted_values = np.sort(list(values))
    yvals = np.arange(1, len(sorted_values) + 1) / float(len(sorted_values))
    
    plt.figure(figsize=(10, 7))
    plt.plot(sorted_values, 1 - yvals, marker='o', linestyle='-', markersize=4)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(title)
    plt.xlabel('Centrality Value')
    plt.ylabel('Fraction of Nodes with Higher Centrality')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.show()

plot_cumulative_distribution(closeness_centrality_values.values(), 'Cumulative Distribution of Closeness Centrality')
