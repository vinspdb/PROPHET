import networkx as nx
import torch_geometric
from dgl.data import DGLDataset


class TextDataset(DGLDataset):
    def __init__(self, X, y):
        self.X_act = X
        self.Y = y

    def __len__(self):
        return len(self.X_act)

    def __plotgraph__(self,data):
            g = torch_geometric.utils.to_networkx(data, to_undirected=False)
            nx.draw(g, with_labels=True)
            import matplotlib.pyplot as plt
            plt.show()
            plt.clf()

    def __getitem__(self, idx):
        data = self.X_act[idx]
        label = self.Y[idx]
        return data, label


