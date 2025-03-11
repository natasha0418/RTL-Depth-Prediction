import pickle

import torch
from torch_geometric.data import DataLoader, Dataset

from dataset import GraphDataset, create_ast_dataset, create_graph_dataset
from model import GCN, train_graph_nn, train_random_forest

create_ast_dataset()
create_graph_dataset()

with open("ast_features.pkl", "rb") as f:
    df = pickle.load(f, encoding="utf-8")

X = df.drop(columns=["filepath", "label"])
y = df["label"]

rf = train_random_forest(X, y)

graph_dataset = GraphDataset()

train_dataset, test_dataset = torch.utils.data.random_split(graph_dataset, [int(0.8 * len(graph_dataset)), len(graph_dataset) - int(0.8 * len(graph_dataset))])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

train_graph_nn(train_loader, test_loader, graph_dataset)

# we also have to add an inverse weighting mechanism here to combine the outputs of both models for optimal performance.
