# -*- coding: utf-8 -*-
import argparse
import numpy as np
import time
import random
import torch
import torch.nn.functional as F
import dgl
from gat import GAT
from dgl.data import CornellDataset, ActorDataset, ChameleonDataset
from torch_geometric.utils import degree, to_undirected
from utils import load_network_data, get_train_data, random_planetoid_splits, aug
from loss import multihead_contrastive_loss
import warnings
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import tqdm
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    MultiLayerFullNeighborSampler,
    negative_sampler,
    NeighborSampler,
)
from dgl.data import AmazonCoBuyComputerDataset

#from time import time

import numpy as np
import pandas as pd


# For plotting
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import plotly.graph_objects as go


sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

#For standardising the dat
from sklearn.preprocessing import StandardScaler

#PCA
from sklearn.manifold import TSNE

#Ignore warnings
import warnings


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GAT')
parser.add_argument("--gpu", type=int, default=0,
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument("--epochs", type=int, default=2000,
                    help="number of training epochs")
parser.add_argument("--dataset", type=str, default="cora",
                    help="which dataset for training")
parser.add_argument("--num-heads", type=int, default=4,
                    help="number of hidden attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--num-hidden", type=int, default=32,
                    help="number of hidden units")
parser.add_argument("--tau", type=float, default=1,
                    help="temperature-scales")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed")
parser.add_argument("--in-drop", type=float, default=0.6,
                    help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=0.5,
                    help="attention dropout")
parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default=1e-4,
                    help="weight decay")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--rate', type=float, default=0.18,
                    help="edge sampling rate")




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.dataset == 'actor':
    dataset = ActorDataset()
    g = dataset[0]
    labels = g.ndata['label'].cpu().numpy()
    n_classes = max(labels).item() + 1
    features = g.ndata['feat']
elif args.dataset == 'chameleon':
    dataset = ChameleonDataset()
    g = dataset[0]
    labels = g.ndata['label'].cpu().numpy()
    n_classes = max(labels).item() + 1
    features = g.ndata['feat']
else:
    adj2, features, Y = load_network_data(args.dataset)
    features[features > 0] = 1
#if args.dataset in ['cora', 'citeseer']:
    g = dgl.from_scipy(adj2)
    features = torch.FloatTensor(features.todense())
    #features = torch.FloatTensor(np.asarray(features.todense()))

    labels = np.argmax(Y, 1)
    #adj = torch.tensor(adj2.todense())
    n_classes = Y.shape[1]
    

#print(labels.shape)
    
gg = aug(g, 1-args.rate)
indic = gg.adj().indices()

line_g = dgl.line_graph(gg)
line_adj = line_g.adj().to_dense()


#else:
    #line_adj = line_g.adj() 

if args.gpu >= 0 and torch.cuda.is_available():
    cuda = True
    g = g.int().to(args.gpu)
else:
    cuda = False

#features = torch.FloatTensor(features.todense())
f = open('AFECL_' + args.dataset + '.txt', 'a+')
f.write('\n\n\n{}\n'.format(args))
f.flush()

#labels = np.argmax(Y, 1)
#adj = torch.tensor(adj2.todense())

all_time = time.time()
num_feats = features.shape[1]
#n_classes = Y.shape[1]
n_edges = g.number_of_edges()

# add self loop
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)



# create model

#if args.dataset in ['cora', 'citeseer']: 
heads = ([args.num_heads] * args.num_layers)
model = GAT(g,
            indic,
            args.num_layers,
            num_feats,
            args.num_hidden,
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.negative_slope)
"""
else:
    heads = ([args.num_heads, args.num_heads] * args.num_layers)
    model = GAT_batch(
            args.num_layers,
            num_feats,
            args.num_hidden,
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.negative_slope)
    g = g.to(device)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    train_nids = torch.arange(g.num_nodes()).to(device)
    dataloader = DataLoader(
      g, train_nids, sampler,
      batch_size=4096,
      shuffle=False,
      drop_last=False,
      num_workers=False)
"""
if cuda:
    model.cuda()
    features = features.cuda()
    #adj = adj.cuda()

# use optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
model.to(device)
# initialize graph

dur = []
test_acc = 0

counter = 0
min_train_loss = 100
early_stop_counter = 500
best_t = -1
min_acc = -1
#print(model)
#labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)

for epoch in range(args.epochs):
    if epoch >= 0:
        t0 = time.time()
    model.train()
    optimizer.zero_grad()
    heads, edge_heads = model(features)
    #if args.dataset in ['cora', 'citeseer']:
    loss = multihead_contrastive_loss(edge_heads, line_adj, tau=args.tau).to(device)
    #else:
        #loss = multihead_contrastive_loss_batch(edge_heads, tau=args.tau, batch_size = 1).to(device)
    if epoch >= 0:
        dur.append(time.time() - t0)
    print("Epoch {:04d} | Time(s) {:.4f} | TrainLoss {:.4f} ".format(epoch + 1, np.mean(dur), loss.item()))
    
    
    loss.backward()
    optimizer.step()


    model.eval()
    with torch.no_grad():
        heads, edge_heads = model(features)
    embeds = torch.cat(heads, axis=1)
    embeds = embeds.detach().cpu()
    Accuaracy_test_allK = []
    numRandom = 20
    train_num = 3
    AccuaracyAll = []
    for random_state in range(numRandom):
        val_num = 0
        #idx_train, idx_val, idx_test = get_train_data(torch.tensor(labels), train_num, val_num, random_state)
        idx_train, idx_val, idx_test = get_train_data(labels, train_num, val_num, random_state)
        train_embs = embeds[idx_train, :]
        val_embs = embeds[idx_val, :]
        test_embs = embeds[idx_test, :]
        LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0)
        LR.fit(train_embs, labels[idx_train])
        y_pred_test = LR.predict(test_embs)  # pred label
        test_acc = accuracy_score(labels[idx_test], y_pred_test)
        AccuaracyAll.append(test_acc)
    average_acc = np.mean(AccuaracyAll) * 100
    std_acc = np.std(AccuaracyAll) * 100
    print('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
        numRandom, average_acc, std_acc, train_num, val_num))
    if min_acc < average_acc:
        counter = 0
        min_acc = average_acc
        best_t = epoch
        torch.save(model.state_dict(), 'best_AFECL.pkl')
    else:
        counter += 1
    if counter >= early_stop_counter:
        print('early stop')
        break
#    model.eval()
#    with torch.no_grad():
#        heads, edge_heads = model(features)
        #if args.dataset in ['cora', 'citeseer']:
#        loss_train = multihead_contrastive_loss(edge_heads, line_adj, tau=args.tau)
        #else:
            #loss_train = multihead_contrastive_loss_batch(edge_heads, tau=args.tau, batch_size = 1).to(device)

    # early stop if loss does not decrease for 100 consecutive epochs
#    if loss_train < min_train_loss:
#        counter = 0
#        min_train_loss = loss_train
#        best_t = epoch
#        torch.save(model.state_dict(), 'best_AFECL.pkl')
#    else:
#        counter += 1

#    if counter >= early_stop_counter:
#        print('early stop')
#        break

#    if epoch >= 0:
#        dur.append(time.time() - t0)

#    print("Epoch {:04d} | Time(s) {:.4f} | TrainLoss {:.4f} ".
#          format(epoch + 1, np.mean(dur), loss_train.item()))

print('Loading {}th epoch'.format(best_t))

model.load_state_dict(torch.load('best_AFECL.pkl'))
model.eval()
with torch.no_grad():
    heads, edge_heads = model(features)
embeds = torch.cat(heads, axis=1)  # concatenate emb learned by all heads
embeds = embeds.detach().cpu()



#tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=50, n_iter=1000).fit_transform(embeds)

#plt.scatter(tsne[:, 0], tsne[:, 1], s= 5, c=labels, cmap='Spectral') 
#plt.gca().set_aspect('equal', 'datalim')
#plt.colorbar(boundaries=np.arange(16)-0.5).set_ticks(np.arange(15))

#plt.savefig('./test2.jpg')
#plt.show()



Accuaracy_test_allK = []
numRandom = 20

for train_num in [1, 2, 3, 4, 20]:

    AccuaracyAll = []
    for random_state in range(numRandom):
        print(
            "\n=============================%d-th random split with training num %d============================="
            % (random_state + 1, train_num))

        if train_num == 20:
            if args.dataset in ['cora', 'citeseer', 'pubmed']:
                # train_num per class: 20, val_num: 500, test: 1000
                val_num = 500
                idx_train, idx_val, idx_test = random_planetoid_splits(n_classes, torch.tensor(labels), train_num,
                                                                       random_state)
            else:
                # Coauthor CS, Amazon Computers, Amazon Photo
                # train_num per class: 20, val_num per class: 30, test: rest
                val_num = 30
                idx_train, idx_val, idx_test = get_train_data(torch.tensor(labels), train_num, val_num, random_state)

        else:
            val_num = 0  # do not use a validation set when the training labels are extremely limited
            idx_train, idx_val, idx_test = get_train_data(torch.tensor(labels), train_num, val_num, random_state)

        train_embs = embeds[idx_train, :]
        val_embs = embeds[idx_val, :]
        test_embs = embeds[idx_test, :]

        if train_num == 20:
            # find the best parameter C using validation set
            best_val_score = 0.0
            for param in [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]:
                LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, C=param)
                LR.fit(train_embs, labels[idx_train])
                val_score = LR.score(val_embs, labels[idx_val])
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_parameters = {'C': param}

            LR_best = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, **best_parameters)

            LR_best.fit(train_embs, labels[idx_train])
            y_pred_test = LR_best.predict(test_embs)  # pred label
            print("Best accuracy on validation set:{:.4f}".format(best_val_score))
            print("Best parameters:{}".format(best_parameters))

        else:  # not use a validation set when the training labels are extremely limited
            LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0)
            LR.fit(train_embs, labels[idx_train])
            y_pred_test = LR.predict(test_embs)  # pred label

        test_acc = accuracy_score(labels[idx_test], y_pred_test)
        print("test accuaray:{:.4f}".format(test_acc))
        AccuaracyAll.append(test_acc)

    average_acc = np.mean(AccuaracyAll) * 100
    std_acc = np.std(AccuaracyAll) * 100
    print('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
        numRandom, average_acc, std_acc, train_num, val_num))
    f.write('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
        numRandom, average_acc, std_acc, train_num, val_num))
    f.flush()

    Accuaracy_test_allK.append(average_acc)

f.close()
