# this is just sample code for visualizing networkx and pytorch data graphs 

# # Read the DOT format graph from a file
# with open("my_circuit.dot", "r") as file:
#     dot_string = file.read()
# # Create a directed graph from the DOT string
# G = nx.nx_agraph.from_agraph(pgv.AGraph(string=dot_string))
# print(G)
# # Map node labels to integer indices
# node_mapping = {node: i for i, node in enumerate(G.nodes)}
# # Convert NetworkX graph to PyTorch Geometric Data format
# edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()
# num_nodes = G.number_of_nodes()
# data = Data(edge_index=edge_index, num_nodes=num_nodes)
# G_pyg = to_networkx(data, to_undirected=True)
# # Draw the graph
# plt.figure(figsize=(10, 6))
# nx.draw(G_pyg, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")
# plt.title("PyTorch Geometric Graph")
# plt.savefig('pytorch_graph.png')
# print(data)
# # Draw the graph
# plt.figure(figsize=(10, 6))
# pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
# nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold")
# plt.title("Logic Circuit Graph")
# plt.savefig("graph.png")  # Save the plot to a file
# print("Graph saved as graph.png")