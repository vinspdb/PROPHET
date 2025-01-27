import pickle
import torch
from dgl.dataloading import GraphDataLoader
from preprocessing.dgl_dataset import TextDataset
from sklearn.metrics import precision_recall_fscore_support
import sys
import h5py


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_graphs_from_hdf5(filename):
        graphs = []
        label = []
        with h5py.File(filename, 'r') as f:
            for key in f.keys():
                pickled_graph = f[key][()]
                graph = pickle.loads(pickled_graph)
                graphs.append(graph['graph'])
                label.append(graph['label'])
        return graphs, label

if __name__ == '__main__':
    log_name = sys.argv[1]
    print(log_name, '-----')

    # Load graphs
    X_test, y_test = load_graphs_from_hdf5('heterographs_tracenode/' + log_name + '_test.db')

    model = torch.load(f'models/model_{log_name}.h5')
    df_test = TextDataset(X_test, y_test)

    test_loader = GraphDataLoader(df_test,
                                  batch_size=256,
                                  drop_last=False,
                                  shuffle=False)

    list_pred = []
    list_truth = []
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            list_edge = X.edges(etype=('concept:name', 'follow', 'concept:name'))
            feature = {}
            for n in X.ntypes:
                feature[n] = X.ndata[n][n]
            model.rgcn(X, feature)
            pred = model(X, feature).argmax(dim=1)
            list_pred.extend(pred.cpu().numpy())
            list_truth.extend(y.argmax(dim=1).cpu().numpy())
    precision, recall, fscore, _ = precision_recall_fscore_support(list_truth, list_pred, average='macro',
                                                                   pos_label=None)
    print("fscore-->{:.3f}".format(fscore))
