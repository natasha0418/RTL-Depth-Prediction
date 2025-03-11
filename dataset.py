import os
import pickle
import signal

import pandas as pd
from graph_features import get_circuit_graph
from torch.utils.data import Dataset
from torch_geometric.data import Dataset

from ast_features import extract_ast_features

class TimeoutException(Exception):
    pass


def handler(signum, frame):
    raise TimeoutException()


def create_ast_dataset(filename):
    df = pd.read_csv(filename)

    feature_list = []
    for idx, row in df.iterrows():
        print(idx, end="---")
        filepath = row["Filepath"]
        level = row["Levels"]

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(180)

        try:
            ast_features = extract_ast_features(filepath)
            ast_features["filepath"] = filepath
            ast_features["label"] = level
            feature_list.append(ast_features)

        except TimeoutException:
            print(f"\n[WARNING] Timeout: Skipping {filepath} (Took longer than 3 minutes)")
            continue
        finally:
            signal.alarm(0)  # Cancel the alarm

    df = pd.DataFrame(feature_list)
    with open("ast_features.pkl", "wb") as f:
        pickle.dump(df, f)
    return df


def create_graph_dataset(filename):
    df = pd.read_csv(filename)

    feature_list = []
    for idx, row in df.iterrows():
        print(idx, end="---")
        filepath = row["Filepath"]
        if not os.path.exists(filepath):
            continue
        level = row["Levels"]

        try:
            graph_features = {}
            graph_features["graph"] = get_circuit_graph(filepath)
            graph_features["filepath"] = filepath
            graph_features["label"] = level
            feature_list.append(graph_features)

        except:
            continue

    df = pd.DataFrame(feature_list)
    with open("graph_features.pkl", "wb") as f:
        pickle.dump(df, f)
    return df


class GraphDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open("graph_features.pkl", "rb") as f:
            self.df = pickle.load(f)
        self.graphs = self.df["graph"].tolist()
        self.labels = self.df["label"].tolist()

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx], self.labels[idx]


# create_graph_dataset('dataset.csv')
