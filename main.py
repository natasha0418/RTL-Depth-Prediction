from ast_features import extract_ast_features
from graph_features import extract_graph_features
from utils import generate_file

from model import train_graph_nn, train_random_forest
file = 'sample_data/file.v'

json_file = "my_circuit.json"
dot_file = "my_circuit.dot"

generate_file(file, 'dot', dot_file)
generate_file(file, 'json', json_file)

# The code for the ML training assumes that labels are provided (combinational depth)
# but for now they are not present.
# While creating the dataset, the combinational depth has to be calculated and provided.

features1, node_mapping, edge_index, df = extract_graph_features(json_file, dot_file)
train_graph_nn(features1, edge_index, df)

features2 = extract_ast_features(file)
train_random_forest(features2)

# we also have to add an inverse weighting mechanism here to combine the outputs of both models for optimal performance.
