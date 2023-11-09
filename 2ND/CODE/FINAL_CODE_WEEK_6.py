# Query 1: Preparing the Graph
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import community as community_louvain

# Load the graph data from CSV files
edges_filename = "edges.csv"
nodes_filename = "nodes.csv"

df_edges = pd.read_csv(edges_filename)
df_nodes = pd.read_csv(nodes_filename)

# Create the graph using the edges CSV
G = nx.from_pandas_edgelist(df_edges, '# source', ' target')

# Ensure the graph is undirected and unweighted
G = nx.to_undirected(G)

# Remove self-loops if necessary
if nx.is_frozen(G):
    G = nx.Graph(G)  # Unfreeze if frozen
G.remove_edges_from(nx.selfloop_edges(G))

# Keep only the largest connected component
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()

# Query 2: Implementing Community Detection Techniques

# 2.1 Bridge Removal Method
def bridge_removal_partition(G):
    # Create a copy of the graph to work with
    G_no_bridges = G.copy()

    # Identify all bridges in the graph
    bridges = list(nx.bridges(G_no_bridges))

    # Remove these bridges from the graph
    G_no_bridges.remove_edges_from(bridges)

    # Identify the connected components in the graph after bridge removal
    components = [G_no_bridges.subgraph(c).copy() for c in nx.connected_components(G_no_bridges)]

    # Initialize lists to store modularity scores and community partitions
    modularity_scores = []
    community_partitions = []

    # Iterate over each component formed after removing bridges
    for comp in components:
        # Get the nodes in the current component
        community = list(comp.nodes)

        # Identify all other nodes in the graph not in the current component
        other_nodes = set(G.nodes()) - set(community)

        # Create a partition dictionary where nodes in the community are marked '0' and others '1'
        partition = {node: 0 for node in community}
        partition.update({node: 1 for node in other_nodes})

        # Convert the partition dictionary to a format suitable for modularity calculation
        reverse_partition = {}
        for node, community_id in partition.items():
            reverse_partition.setdefault(community_id, set()).add(node)

        # Get the communities as a list of sets of nodes
        communities = list(reverse_partition.values())

        # Store the partition
        community_partitions.append(partition)

        # Calculate and store the modularity score for this partition
        modularity_score = nx.community.modularity(G, communities)
        modularity_scores.append(modularity_score)

    # Identify the partition with the highest modularity score
    max_modularity_index = modularity_scores.index(max(modularity_scores))

    # Return the partition with the highest modularity
    return community_partitions[max_modularity_index]



# Apply Bridge Removal
start_time = time.time()
best_partition_bridge_removal = bridge_removal_partition(G)
bridge_time = time.time() - start_time

# 2.2 Modularity Optimization
start_time = time.time()
partition_modularity_optimization = community_louvain.best_partition(G)
modularity_time = time.time() - start_time

# 2.3 Label Propagation
start_time = time.time()
communities_label_prop = list(nx.community.label_propagation_communities(G))
label_prop_time = time.time() - start_time
partition_label_prop = {node: i for i, comm in enumerate(communities_label_prop) for node in comm}

# Query 3: Analysis and Comparison of Results
def collect_results(G, partition):
    # Convert partition to a list of sets of nodes
    reverse_partition = {}
    for node, community_id in partition.items():
        reverse_partition.setdefault(community_id, set()).add(node)
    
    communities = list(reverse_partition.values())
    num_clusters = len(communities)
    cluster_sizes = [len(c) for c in communities]
    modularity = nx.community.modularity(G, communities)
    return num_clusters, cluster_sizes, modularity


# Collecting results
bridge_results = collect_results(G, best_partition_bridge_removal)
modularity_results = collect_results(G, partition_modularity_optimization)
label_prop_results = collect_results(G, partition_label_prop)

# Creating a DataFrame for comparison
results_df = pd.DataFrame({
    'Method': ['Bridge Removal', 'Modularity Optimization', 'Label Propagation'],
    'Number of Clusters': [bridge_results[0], modularity_results[0], label_prop_results[0]],
    'Cluster Size Distribution': [bridge_results[1], modularity_results[1], label_prop_results[1]],
    'Computational Time (s)': [bridge_time, modularity_time, label_prop_time],
    'Modularity': [bridge_results[2], modularity_results[2], label_prop_results[2]]
})

print(results_df)

# Query 4: Visualization Hints (To be done in Gephi)
# Choose the best partition based on your analysis (e.g., partition_modularity_optimization)
best_partition = partition_modularity_optimization

# Add community information to nodes for export
for node, comm_id in best_partition.items():
    G.nodes[node]['community'] = comm_id

# Export the graph with community data to GEXF for Gephi
nx.write_gexf(G, "network_with_communities.gexf")
