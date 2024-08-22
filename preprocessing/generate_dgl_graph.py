import sys
import numpy as np
import pandas as pd
import torch
from preprocessing.read_event_log import GenerateTrace
from collections import OrderedDict
from sklearn.preprocessing import OneHotEncoder
import pickle
from utility import log_config as lc
import dgl
from sklearn import preprocessing
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from Orange.data.pandas_compat import table_from_frame
import Orange
from dgl import AddReverse
import sqlite3

def replace_char(ele):
    ele = ele.replace(' ', '')
    ele = ele.replace('-', '')
    ele = ele.replace('+', '')
    ele = ele.replace('_', '')
    ele = ele.replace('.', '')
    ele = ele.replace(':', '')
    ele = ele.replace('(', '')
    ele = ele.replace(')', '')
    return ele

def build_w2v(prefix_list, mean):
        temp_traces = []
        for k in prefix_list:
            listToStr = ' '.join([replace_char(str(elem)) for elem in k])
            temp_traces.append(listToStr)

        tokenized_words = []
        for s in temp_traces:
            tokenized_words.append(s.split(' '))

        w2v_model = Word2Vec(vector_size=mean, seed=0, min_count=1, sg=0, workers=-1)
        w2v_model.build_vocab(tokenized_words, min_count=1)
        total_examples = w2v_model.corpus_count
        w2v_model.train(tokenized_words, total_examples=total_examples, epochs=50)
        vocab = list(w2v_model.wv.index_to_key)
        word_vec_dict = {}
        for word in vocab:
            word_vec_dict[word] = w2v_model.wv.get_vector(word)
        return word_vec_dict

def clear_list(prefix_list):
    temp_traces = []
    for k in prefix_list:
        listToStr = ' '.join([replace_char(str(elem)) for elem in k])
        temp_traces.append(listToStr)

    tokenized_words = []
    for s in temp_traces:
        tokenized_words.append(s.split(' '))
    return tokenized_words


def gen_flow(id):
    id.insert(0, 'START')
    remove_dup = list(dict.fromkeys(id))
    remove_dup = [[a] for a in remove_dup]
    id = np.array(id)
    node_encoder = preprocessing.LabelEncoder()
    enc = node_encoder.fit_transform(id)

    return remove_dup, enc

def gen_edge_weigts(list1, list2):
    combined_tuples = list(zip(list1, list2))
    tuple_counts = {}
    for pair in combined_tuples:
        if pair in tuple_counts:
            tuple_counts[pair] += 1
        else:
            tuple_counts[pair] = 1
    return list(tuple_counts.values())


def apply_w2v(list_act, enc_act, mean):
    list_act = clear_list(list_act)
    x_act_ohe = []
    for l in list_act:
        list_emb_temp = []
        for t in l:
            embed_vector = enc_act.get(t)
            if embed_vector is not None:
                list_emb_temp.append(embed_vector)
            else:
                list_emb_temp.append(np.zeros(shape=(mean,)))
        x_act_ohe.append(list_emb_temp)
    x_act_ohe = np.array(x_act_ohe)
    x_act_ohe = x_act_ohe.reshape(x_act_ohe.shape[0], mean)
    return x_act_ohe


def unique_edge(list1, list2):
    unique_tuples = []
    seen_tuples = set()

    for pair in zip(list1, list2):
        if pair not in seen_tuples:
            unique_tuples.append(pair)
            seen_tuples.add(pair)

    return unique_tuples


def build_list_graphs(dict_view, dict_enc, mean, c):
        k = 0
        while k<len(dict_view['activity']):
                list_node = {}
                list_node_comp = {}
                list_node_feature = {}
                dgl_canonical_edge = {}
                weight_node_follow_node = {}
                for v in lc.dict_log[log_name]['event_attr']:
                    list_node[v], list_node_comp[v] = gen_flow(dict_view[v][k])
                    list_node_feature[v] = apply_w2v(list_node[v], dict_enc[v], mean) #W2W

                list_att_trace = []
                for v in lc.dict_log[log_name]['trace_attr_cat']:
                    embed_vector = dict_enc[v].get(replace_char(dict_view[v][k][0]))
                    if embed_vector is not None:
                        res = embed_vector
                    else:
                        res = np.zeros(shape=(mean,))
                    list_att_trace.append(res)

                for v in lc.dict_log[log_name]['trace_attr_num']:
                    list_att_trace.append(np.array(dict_view[v][k][0]).reshape(1))
                if list_att_trace !=[]:
                    list_node_comp['trace_att'] = [0]
                    list_node_feature['trace_att'] = np.array([np.concatenate(list_att_trace)])

                for rel in lc.dict_log[log_name]['relation']:
                    if rel[1] == 'follow':
                        edge_res = np.array([[list_node_comp[rel[0]][i], list_node_comp[rel[0]][i + 1]] for i in range(len(list_node_comp[rel[0]]) - 1)])
                    elif rel[1] == 'has_ta':
                        list_node_comp[rel[2]] = [0]*len(np.unique(list_node_comp[rel[0]]))
                        edge_res = list(map(lambda X: [X[0], X[1]], list(zip(np.unique(list_node_comp[rel[0]]), list_node_comp[rel[2]]))))
                    else:
                        edge_res = list(map(lambda X: [X[0], X[1]], list(zip(list_node_comp[rel[0]], list_node_comp[rel[2]]))))
                    src = [item[0] for item in edge_res]
                    dst = [item[1] for item in edge_res]
                    tuple_src_dst = unique_edge(src, dst)
                    dgl_canonical_edge[rel] = tuple_src_dst
                    weight_node_follow_node[rel] = gen_edge_weigts(src,dst)
                hetero_graph = dgl.heterograph(dgl_canonical_edge)


                for nn in list_node_feature:
                    hetero_graph.nodes[nn].data[nn] = torch.tensor(list_node_feature[nn], dtype=torch.float)

                for rel in weight_node_follow_node:
                    hetero_graph.edata['h'] = {rel:torch.tensor(weight_node_follow_node[rel])}
                transform = AddReverse(copy_edata=True)
                new_g = transform(hetero_graph)
                insert_graph(new_g, c)
                k = k + 1


def equifreq(view_train, view_test, n_bin):
        sort_v = np.append(view_train, view_test)
        df = pd.DataFrame(sort_v)
        df = table_from_frame(df)
        disc = Orange.preprocess.Discretize()
        disc.method = Orange.preprocess.discretize.EqualFreq(n=n_bin)
        df = disc(df)
        df = list(df)
        df = list(map(str, df))
        view_train = df[:len(view_train)]
        view_test = df[len(view_train):]
        return view_train, view_test

def add_time_column(group):
        timestamp_col = 'timestamp'
        group = group.sort_values(timestamp_col, ascending=True)
        # end_date = group[timestamp_col].iloc[-1]
        start_date = group[timestamp_col].iloc[0]

        timesincelastevent = group[timestamp_col].diff()
        timesincelastevent = timesincelastevent.fillna(pd.Timedelta(seconds=0))
        group["timesincelastevent"] = timesincelastevent.apply(
            lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds

        elapsed = group[timestamp_col] - start_date
        elapsed = elapsed.fillna(pd.Timedelta(seconds=0))
        group["timesincecasestart"] = elapsed.apply(lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds

        return group

def insert_graph(graph, c):
            serialized_graph = pickle.dumps(graph)
            c.execute('''INSERT INTO graphs (graph) VALUES (?)''', (serialized_graph,))
            conn.commit()


if __name__ == '__main__':

        log_name = sys.argv[1]

        print('log-->',log_name)

        pm = GenerateTrace(log_name)
        log = pm.import_log(log_name)
        mean = 100

        log = log.fillna('unk')
        log_train, log_test = pm.generate_split(log)

        log_train['timestamp'] = pd.to_datetime(log_train['timestamp'])
        log_test['timestamp'] = pd.to_datetime(log_test['timestamp'])

        log_train = log_train.groupby('case', group_keys=False).apply(add_time_column)
        log_test = log_test.groupby('case', group_keys=False).apply(add_time_column)
        log_train = log_train.reset_index(drop=True)
        log_test = log_test.reset_index(drop=True)


        if log_name != 'helpdesk2' and log_name != 'sp2020':
            num_act = list(set(list(log_train['activity'].unique()) + list(log_test['activity'].unique())))
            num_res = list(set(list(log_train['resource'].unique()) + list(log_test['resource'].unique())))
            n_bin = (len(num_act) + len(num_res)) // 2

            log_train['resource'] = log_train['resource'].astype(str)
            log_test['resource'] = log_test['resource'].astype(str)
        else:
            num_act = list(set(list(log_train['activity'].unique()) + list(log_test['activity'].unique())))
            n_bin = len(num_act)

        log_train['timesincecasestart'], log_test['timesincecasestart'] = equifreq(log_train['timesincecasestart'], log_test['timesincecasestart'], n_bin)

        dict_card = {}
        dict_view_train = {}
        dict_view_test = {}
        dict_view_train_y = {}
        dict_view_test_y = {}
        dict_enc = {}

        for attr in lc.dict_log[log_name]['event_attr']:
            node_encoder = preprocessing.OneHotEncoder()
            dict_card[attr] = list(set(pd.concat([log_train[attr],log_test[attr]])))
            dict_card[attr].insert(0, 'START')
            dict_view_train[attr], dict_view_train_y[attr] = pm.get_sequence(pm.generate_prefix_trace(log=log_train, view=attr))
            dict_view_test[attr], dict_view_test_y[attr] = pm.get_sequence(pm.generate_prefix_trace(log=log_test, view=attr))
            dict_enc[attr] = build_w2v(dict_view_train[attr], mean)
            output = open("w2v/" + log_name + "/" + attr + ".pkl", 'wb')
            pickle.dump(dict_enc[attr], output)
            output.close()
            dict_att = [[a] for a in dict_card[attr]]

        for attr in lc.dict_log[log_name]['trace_attr_num']:
            scaler = MinMaxScaler()
            log_train[attr] = scaler.fit_transform(log_train[attr].values.reshape(len(log_train),-1))
            log_test[attr] = scaler.transform(log_test[attr].values.reshape(len(log_test),-1))
            dict_view_train[attr] = pm.get_sequence_num(pm.generate_prefix_trace(log=log_train, view=attr))
            dict_view_test[attr] = pm.get_sequence_num(pm.generate_prefix_trace(log=log_test, view=attr))


        for attr in lc.dict_log[log_name]['trace_attr_cat']:
            dict_card[attr] = list(set(pd.concat([log_train[attr], log_test[attr]])))
            dict_view_train[attr] = pm.get_sequence_num(pm.generate_prefix_trace(log=log_train, view=attr))
            dict_view_test[attr] = pm.get_sequence_num(pm.generate_prefix_trace(log=log_test, view=attr))
            dict_enc[attr] = build_w2v(dict_view_train[attr], mean)
            output = open("w2v/" + log_name + "/" + attr + ".pkl", 'wb')
            pickle.dump(dict_enc[attr], output)
            output.close()
            dict_att = [[a] for a in dict_card[attr]]

        conn = sqlite3.connect('heterographs_tracenode/' + log_name + '_train.db')
        c = conn.cursor()

        # Create table if not exists
        c.execute('''CREATE TABLE IF NOT EXISTS graphs (id INTEGER PRIMARY KEY, graph BLOB)''')

        build_list_graphs(dict_view_train, dict_enc, mean, c)
        print('end_train')
        conn.close()

        conn = sqlite3.connect('heterographs_tracenode/' + log_name + '_test.db')
        c = conn.cursor()

        # Create table if not exists
        c.execute('''CREATE TABLE IF NOT EXISTS graphs (id INTEGER PRIMARY KEY, graph BLOB)''')

        build_list_graphs(dict_view_test, dict_enc, mean, c)

        label_encoder = preprocessing.LabelEncoder()
        dict_card['activity'].remove('START')

        integer_encoded = label_encoder.fit_transform(dict_card['activity'])
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        onehot_encoder.fit(integer_encoded)
        onehot_encoded = onehot_encoder.transform(integer_encoded)

        train_integer_encoded = label_encoder.transform(dict_view_train_y['activity']).reshape(-1, 1)
        train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)
        Y_train = np.asarray(train_onehot_encoded)

        test_integer_encoded = label_encoder.transform(dict_view_test_y['activity']).reshape(-1, 1)
        test_onehot_encoded = onehot_encoder.transform(test_integer_encoded)
        Y_test = np.asarray(test_onehot_encoded)
        Y_test_int = np.asarray(test_integer_encoded)

        with open('heterographs_tracenode/'+log_name+'_ytrain.pickle', 'wb') as handle:
            pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('heterographs_tracenode/'+log_name+'_ytest.pickle', 'wb') as handle:
            pickle.dump(Y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('heterographs_tracenode/'+log_name+'_ytestint.pickle', 'wb') as handle:
            pickle.dump(Y_test_int, handle, protocol=pickle.HIGHEST_PROTOCOL)
