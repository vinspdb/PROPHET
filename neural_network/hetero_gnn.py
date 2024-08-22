from neural_network.gatv2_nn import RGCN
import torch.nn as nn
import dgl

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, params, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, params, rel_names)
        self.classify = nn.Linear(params['hidden_dim'], n_classes)

    def forward(self, graph, feat, eweight=None):

        h = self.rgcn(graph, feat)
        with graph.local_scope():
            graph.ndata['h'] = h
            hg = 0
            for ntype in graph.ntypes:
                hg = hg + dgl.sum_nodes(graph, 'h', ntype=ntype)
            return self.classify(hg)


