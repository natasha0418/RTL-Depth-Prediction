from ast_features import extract_ast_features
from graph_features import extract_graph_features
from utils import generate_file

file = 'sample_data/file.v'

json_file = "my_circuit.json"
dot_file = "my_circuit.dot"

generate_file(file, 'dot', dot_file)
generate_file(file, 'json', json_file)

features1, node_mapping, edge_index, df = extract_graph_features(json_file, dot_file)

features2 = extract_ast_features(file)

