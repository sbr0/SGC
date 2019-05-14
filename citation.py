import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed, ssgc_precompute
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter
import json
import struct

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

adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)

torch.set_printoptions(threshold=100000)
print("ADJ: ", sum(adj.to_dense()[1]))
print("ADJ size: ", adj.to_dense().size())
print("features: ", features.size())
print("labels: ", labels.size())
#print("idx_train: ", idx_train)
#print("idx_val: ", idx_val)
#print("idx_test: ", idx_test)
#with open('adj.json', 'w') as outfile:
#    json.dump(adj.to_dense().numpy().tolist(), outfile)

#with open('idx_val.json', 'w') as outfile:
#    json.dump(idx_val.numpy().tolist(), outfile)

#with open('idx_test.json', 'w') as outfile:
#    json.dump(idx_test.numpy().tolist(), outfile)

model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)


new_features = torch.empty_like(features)
#new_features, precomp2 = ssgc_precompute(features, adj, args.degree)
if args.model == "SGC": features, precompute_time = sgc_precompute(features, adj, args.degree)
print("{:.4f}s".format(precompute_time))

# print result of precomputation
with open('python_precomp.bin', 'wb') as outfile:
    graph_size = features.size()[0]
    feat_size = features.size()[1]
    for i in range(graph_size):
        for j in range(feat_size):
            outfile.write(bytearray(struct.pack("f",features[i][j].numpy())))


#print(torch.all(torch.eq(new_features, features)))


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
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time

def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)


if args.model == "SGC":
    model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                     args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test = test_regression(model, features[idx_test], labels[idx_test])



    test_res = torch.mm(features[idx_test], torch.transpose(model.W.weight.data, 0, 1))
    print("DIY test res", test_res.size())
    print("model test res: " , model(features[idx_test]).size())
    #print("DIFF: ", torch.eq(test_res, model(features[idx_test])))

    print(list(model.parameters())[1])
    for param in model.parameters():
        print(type(param.data), param.size())

print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
