import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
import math
import struct

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #print(adj)
    # What is the point of this? Gives same result
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print(adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]


    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)


    adj1 = sparse_mx_to_torch_sparse_tensor(adj).to_dense() # A
    SIZE = adj1.size()[0]
    adj1 = adj1.add(torch.eye(SIZE)) # A^
    print("^A: ", adj1[4])
    adj2 = torch.sum(adj1, dim=1)
    adj3 = torch.zeros(SIZE, SIZE)

    for i in range(SIZE):
        adj3[i][i] = adj2[i]
    adj3 = torch.inverse(adj3.sqrt())
    print("^D^-1/2: ", adj3)
    MANUAL_adj = torch.mm(adj3, torch.mm(adj1, adj3))
    print("S: ", MANUAL_adj)
    # This is where normalisation happens
    adj, features = preprocess_citation(adj, features, normalization)
    OG_adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
    print("OG == MANUAL: ", torch.all(torch.eq(OG_adj, MANUAL_adj)))


    # print degree matrix to file using float format
    # with open('python_degree.bin', 'wb') as outfile:
    #     for i in range(SIZE):
    #         outfile.write(struct.pack('f', adj3.numpy()[i][i]))

    # # print normalised adjacency matrix with self loops
    # with open('python_norm_adj.bin', 'wb') as outfile:
    #     outfile.write(SIZE.to_bytes(4, byteorder='little'))
    #     for i in range(SIZE):
    #         neighbour_count = 0
    #         neighbour_list = list()
    #         for j in range(SIZE):
    #             if OG_adj[i][j] != 0:
    #                 neighbour_count += 1
    #                 neighbour_list.append(j)
    #         neighbour_list.sort()
    #         outfile.write(neighbour_count.to_bytes(4, byteorder='little'))
    #         for j in neighbour_list:
    #             outfile.write(j.to_bytes(4, byteorder='little'))
    #             outfile.write(struct.pack('f', OG_adj.numpy()[i][j]))


    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # # print(graph)
    # # print sparce adjacency matrix to file using uint32_t format
    # with open('sparce.bin', 'wb') as outfile:
    #     graph_size = features.size()[0]
    #     outfile.write(graph_size.to_bytes(4, byteorder='little'))
    #     for i in range(graph_size):
    #         graph[i] = list(set(graph[i])) # Remove duplicates. Necessary!
    #         graph[i].sort() # Not necessary but makes more sense?
    #         num_neighbors = len(graph[i])
    #         outfile.write(num_neighbors.to_bytes(4, byteorder='little'))
    #         for j in range(num_neighbors):
    #             outfile.write(graph[i][j].to_bytes(4, byteorder='little'))


    return adj, features, labels, idx_train, idx_val, idx_test

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def ssgc_precompute(features, adj, degree):
    t = perf_counter()
    adj = adj.to_dense()
    for d in range(degree):
        print("Degree: ", d)
        new_features = torch.empty_like(features)
        for i in range(list(adj.size())[0]):
            print("Outer:", i)
            for j in range(list(features.size())[1]):
                print("Inner", j)
                new_features[i][j] = 0
                for k in range(list(adj.size())[1]):
                    new_features[i][j] += adj[i][k] * features[k][j]
    precompute_time = perf_counter()-t
    return new_features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']

def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index
