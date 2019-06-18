import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed, ssgc_precompute, sparse_mx_to_torch_sparse_tensor
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter
import json
import struct

import scipy as sp
# import cPickle
import gzip

# Arguments
args = get_citation_args()

if args.tuned:
    if args.model == "SGC":
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented

# setting random seeds
set_seed(args.seed, args.cuda)

# adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)

def load_obj(filename, serializer=pkl):
    with gzip.open(filename, 'rb') as fin:
        obj = serializer.load(fin, encoding='latin1')
    return obj

data = load_obj('geotext.pkl')

adj, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = data

#A is the normalised laplacian matrix as A_hat in Kipf et al. (2016).
#The X_? and Y_? should be concatenated to be feed to GCN.
features = sp.sparse.vstack([X_train, X_dev, X_test])
if len(Y_train.shape) == 1:
    labels = np.hstack((Y_train, Y_dev, Y_test))
else:
    labels = np.vstack((Y_train, Y_dev, Y_test))

#get train/dev/test indices in X, Y, and A.
idx_train = np.asarray(range(0, X_train.shape[0])).astype(np.int32)
idx_val = np.asarray(range(X_train.shape[0], X_train.shape[0] + X_dev.shape[0])).astype(np.int32)
idx_test = np.asarray(range(X_train.shape[0] + X_dev.shape[0], X_train.shape[0] + X_dev.shape[0] + X_test.shape[0])).astype(np.int32)

adj = sparse_mx_to_torch_sparse_tensor(adj)

labels = torch.from_numpy(labels).long()
features = torch.from_numpy(features.toarray()).float()
# features = sparse_mx_to_torch_sparse_tensor(features)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

print("ADJ: ", adj)
print("features: ", features)
print("labels: ", labels)

torch.set_printoptions(threshold=100000)
# print("ADJ: ", sum(adj.to_dense()[1]))
# print("ADJ size: ", adj.to_dense().size())
# print("features: ", features.size())
print("labels: ", labels.size())
# print("idx_train: ", idx_train)
# print("idx_val: ", idx_val)
#print("idx_test: ", idx_test)
#with open('adj.json', 'w') as outfile:
#    json.dump(adj.to_dense().numpy().tolist(), outfile)

#with open('idx_val.json', 'w') as outfile:
#    json.dump(idx_val.numpy().tolist(), outfile)

#with open('idx_test.json', 'w') as outfile:
#    json.dump(idx_test.numpy().tolist(), outfile)

model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

# # print feature matrix
# with open('features.bin', 'wb') as outfile:
#     graph_size = features.size()[0]
#     feat_size = features.size()[1]
#     for i in range(graph_size):
#         for j in range(feat_size):
#             outfile.write(bytearray(struct.pack("f",features[i][j].numpy())))

# # Compressed sparse row (CSR) format
# with open('CSR_values.bin', 'wb') as V, \
#         open('CSR_Idx.bin', 'wb') as Idx, \
#         open('CSR_Ptr.bin', 'wb') as Ptr:
#     graph_size = features.size()[0]
#     feat_size = features.size()[1]
#     k = 0
#     for i in range(graph_size):
#         Ptr.write(bytearray(struct.pack("I", k)))
#         for j in range(feat_size):
#             if (features[i][j] != 0):
#                 V.write(bytearray(struct.pack("f",features[i][j].numpy())))
#                 Idx.write(bytearray(struct.pack("I", j)))
#                 k = k + 1


# print(features)

# # print labels matrix
# with open('labels.bin', 'wb') as outfile:
#     graph_size = labels.size()[0]
#     for i in range(graph_size):
#         outfile.write(bytearray(struct.pack("I",labels[i].numpy())))

new_features = torch.empty_like(features)
#new_features, precomp2 = ssgc_precompute(features, adj, args.degree)
if args.model == "SGC": features, precompute_time = sgc_precompute(features, adj, args.degree)
print("{:.4f}s".format(precompute_time))

# print(features)
# # print result of precomputation
# with open('python_precomp.bin', 'wb') as outfile:
#     graph_size = features.size()[0]
#     feat_size = features.size()[1]
#     for i in range(graph_size):
#         for j in range(feat_size):
#             outfile.write(bytearray(struct.pack("f",features[i][j].numpy())))


#print(torch.all(torch.eq(new_features, features)))

# print starting weights (aiming distribution [-sqrt(1/d), sqrt(1/d)])

# with open('python_starting_weights.bin', 'wb') as outfile:
#     feat_size = features.size()[1]
#     classification_size = labels.max().item()+1
#     for i in range(classification_size):
#         for j in range(feat_size):
#             outfile.write(bytearray(struct.pack("f",list(model.parameters())[0][i][j].detach().numpy())))


# print starting biases

# with open('python_starting_biases.bin', 'wb') as outfile:
#     classification_size = labels.max().item()+1
#     for i in range(classification_size):
#         outfile.write(bytearray(struct.pack("f",list(model.parameters())[1][i].detach().numpy())))

# https://deepnotes.io/softmax-crossentropy
# https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/
def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Inference / passe-avant => NxC
        output = model(train_features)
        # if epoch == 0:
        #     print("INFERED:", output)

        # if epoch == 0:
        #     print("TRAIN FEAT SIZE: ", train_features[0])
            # print("OUTPUT SIZE: ", output.size())
            # print("OUTPUT: ", output[0])
            # print("WEIGHTS: ", list(model.parameters())[0][0])
        # softmax + Loss function => SCALAR !?
        loss_train = F.cross_entropy(output, train_labels)
        # if epoch == 0:
            # print("LOSS TRAIN SIZE: ", loss_train.size())
        if epoch == 0 or epoch == epochs -1:
            print("LOSS TRAIN: ", loss_train)
        loss_train.backward()
        if epoch == 0:
            print (optimizer.state_dict())
        optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        # print("VAL_LABELS:", output)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time

def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)


if args.model == "SGC":
    # print("start BIASES", list(model.parameters())[1])
    model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                     args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test = test_regression(model, features[idx_test], labels[idx_test])



    test_res = torch.mm(features[idx_test], torch.transpose(model.W.weight.data, 0, 1))
    print("DIY test res", test_res.size())
    print("model test res: " , model(features[idx_test]).size())
    #print("DIFF: ", torch.eq(test_res, model(features[idx_test])))

    print("end BIASES", list(model.parameters())[1])
    for param in model.parameters():
        print(type(param.data), param.size())

print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
