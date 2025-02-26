import json
import re

import networkx as nx
import pandas as pd
import torch


def extract_gate_info(label):
    pattern = r"\$(\d+)\s*(?:\\n|\n)\s*\$(\w+)"
    match = re.search(pattern, label)
    if match:
        gate_index = int(match.group(1))  # Extract gate index (e.g., 1 or 5)
        gate_name = match.group(2)  # Extract gate name (e.g., 'and' or 'or')
        return gate_name, gate_index
    return None, None


def extract_io_nodes(json_file, dot_file):
    """
    Extracts input and output nodes from the DOT file and maps them using JSON.

    Args:
        json_file (str): Path to the JSON file from Yosys.
        dot_file (str): Path to the DOT file.

    Returns:
        dict: Mapping from DOT node names to logical names.
        dict: Dictionary containing input/output nodes.
        nx.DiGraph: Parsed NetworkX graph.
    """
    with open(json_file, "r") as f:
        yosys_data = json.load(f)

    G = nx.drawing.nx_agraph.read_dot(dot_file)
    module = next(iter(yosys_data["modules"].values()))
    ports = module["ports"]

    dot_to_logical = {}  # Mapping DOT nodes (n5, n6, n7) to logical names (a, b, y)
    io_nodes = {"inputs": set(), "outputs": set()}  # To store inputs/outputs separately

    for node, data in G.nodes(data=True):
        if "label" in data:
            label = data["label"].strip('"')
            if label in ports:
                dot_to_logical[node] = label
                if ports[label]["direction"] == "input":
                    io_nodes["inputs"].add(node)
                elif ports[label]["direction"] == "output":
                    io_nodes["outputs"].add(node)

    return dot_to_logical, io_nodes, G


def extract_gates(G):
    """
    Extracts unique gate types from the DOT graph for one-hot encoding.

    Args:
        G (nx.DiGraph): NetworkX graph parsed from DOT.

    Returns:
        list: Sorted list of unique gate types.
    """
    gates = {}

    for node in G.nodes():
        if node.startswith("c"):  # Gate nodes typically start with 'c'
            label = G.nodes[node].get("label", "")
            gate_name, gate_index = extract_gate_info(label)
            if gate_name:
                gates[gate_index] = gate_name
    return gates


def assign_features(G, io_nodes, gates):
    """
    Builds a feature matrix using input/output classification and gate type encoding.

    Args:
        G (nx.DiGraph): NetworkX graph.
        io_nodes (dict): Dictionary containing input/output nodes.
        gates (dict): Unique gate types from the DOT file.

    Returns:
        torch.Tensor: Node feature matrix.
        dict: Mapping from node names to indices.
        torch.Tensor: Edge index tensor.
        pd.DataFrame: Human-readable DataFrame of features.
    """
    print(gates)
    gate_types = gates.values()

    df = pd.DataFrame(0, index=G.nodes(), columns=sorted(set(gate_types)) + ["is_input", "is_output"])

    for node in G.nodes():
        if node in io_nodes["inputs"]:
            df.at[node, "is_input"] = 1
        elif node in io_nodes["outputs"]:
            df.at[node, "is_output"] = 1

        if node.startswith("c"):  # It's a gate node
            print(G.nodes[node]["label"])
            gate_type, gate_index = extract_gate_info(G.nodes[node]["label"])
            df.at[node, gate_type] = 1  # Mark the correct gate type

    node_mapping = {node: i for i, node in enumerate(G.nodes())}

    x = torch.tensor(df.values, dtype=torch.float)
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges()], dtype=torch.long).t().contiguous()

    return x, node_mapping, edge_index, df


def process_graph(json_file, dot_file):
    """
    Full pipeline: Extracts nodes, gates, and assigns features for GNN processing.

    Args:
        json_file (str): Path to the JSON file.
        dot_file (str): Path to the DOT file.

    Returns:
        torch.Tensor: Node feature matrix.
        dict: Node mapping.
        torch.Tensor: Edge index tensor.
        pd.DataFrame: DataFrame of features.
    """
    dot_to_logical, io_nodes, G = extract_io_nodes(json_file, dot_file)
    gates = extract_gates(G)
    x, node_mapping, edge_index, df = assign_features(G, io_nodes, gates)

    return x, node_mapping, edge_index, df
