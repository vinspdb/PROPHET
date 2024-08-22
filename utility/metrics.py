import pickle
import torch
from dgl.dataloading import GraphDataLoader
from preprocessing.dgl_dataset import TextDataset
from sklearn.metrics import precision_recall_fscore_support
import sys
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import sqlite3

def fetch_graphs():
    c.execute('''SELECT graph FROM graphs''')
    rows = c.fetchall()
    graph_list = []
    for row in rows:
        serialized_graph = row[0]
        graph = pickle.loads(serialized_graph)
        graph_list.append(graph)
    return graph_list

if __name__ == '__main__':
    log_name = sys.argv[1]
    print(log_name, '-----')

    conn = sqlite3.connect("heterographs_tracenode/" + log_name + '_test.db')
    c = conn.cursor()

    X_test = fetch_graphs()
    
    with open(f'heterographs_tracenode/{log_name}_ytest.pickle',
              'rb') as handle:
        y_test = pickle.load(handle)

    with open(f'heterographs_tracenode/{log_name}_ytestint.pickle',
              'rb') as handle:
        y_test_int = pickle.load(handle)

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
            list_edge = X.edges(etype=('activity', 'follow', 'activity'))
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
