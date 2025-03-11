import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


def train_random_forest(X, y):
    # X = features.drop(columns=['label'])
    # y = features['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Random Forest Regression MSE: {mse:.4f}, R²: {r2:.4f}")

    return rf


def train_graph_nn(train_loader, test_loader, graph_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GCN(in_features=graph_list[0].x.shape[1], hidden_dim=64, out_features=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()  # Regression loss

    # Training loop
    for epoch in range(100):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            output = model(batch.x, batch.edge_index, batch.batch)  # Predict depth
            loss = criterion(output, batch.y)  # Compute MSE loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")

        model.eval()

    test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")


class GCN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Fully connected layer for prediction
        self.fc = nn.Linear(hidden_dim, out_features)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)

        x = self.fc(x)  # Final prediction
        return x.squeeze()  # Output a single value per graph


# Training loop for the GNN.
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred == data.y).sum().item()
    acc = correct / data.y.size(0)
    return acc


# Train for a number of epochs
