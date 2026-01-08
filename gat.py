"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self,
                 g,
                 indics : torch.tensor,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope):
        super(GAT, self).__init__()
        self.g = g
        self.indics = indics
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, False, self.activation))
        #self.fc1 = torch.nn.Linear(2 * num_hidden * heads[0], 2 *num_hidden * heads[0])
        #self.fc2 = torch.nn.Linear(2 * num_hidden * heads[0], 2 *num_hidden * heads[0])
        
    def forward(self, inputs):
        heads = []
        edge_heads = []
        h = inputs
        # get hidden_representation
        for l in range(self.num_layers):
            temp = h.flatten(1)
            h =self.gat_layers[l](self.g, temp)
        edge_h = torch.cat((h[self.indics[0, :], :], h[self.indics[1, :], :]), dim=2)
        # get heads
        for i in range(h.shape[1]):
            heads.append(h[:, i])
            edge_heads.append(edge_h[:, i])
        edges = torch.cat(edge_heads, axis = 1)
        print(edges.shape)
        #edges = F.elu(self.fc1(edges))
        #edges = self.fc2(edges)
        tmp = []
        tmp.append(edges)
        
        return heads, tmp


            
            
            
            
            
            
            
            
            
            
            
            
    