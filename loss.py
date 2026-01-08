# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from typing import Optional
import dgl
import numpy as np

def contrastive_loss(z1: torch.Tensor, adj: torch.Tensor,
                     mean: bool = True, tau: float = 1.0):

    l1 = nei_con_loss(z1, adj, tau)
    #l2 = nei_con_loss(z2, z1, adj, tau)
    ret = l1
    ret = ret.mean() if mean else ret.sum()

    return ret



def multihead_contrastive_loss(heads, adj: torch.Tensor, tau: float = 1.0):
    loss = torch.tensor(0, dtype=float, requires_grad=True)
    for i in range(0, len(heads)):
        loss = loss + contrastive_loss(heads[0], adj, tau=tau)

            
    #return loss / (len(heads) - 1)
    return loss




def sim(z1: torch.Tensor, z2: torch.Tensor):

    #z1 = F.normalize(z1)
    #z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def nei_con_loss(z1: torch.Tensor, adj: torch.Tensor, tau):
    '''neighbor contrastive loss'''
    adj = adj - torch.diag_embed(adj.diag())  # remove self-loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj[adj > 0] = 1
    
    #n_class = adj.shape[0]
    #L = np.zeros((n_class, n_class))
    #L[np.tril_indices(n_class)] = 1.
    #L =torch.from_numpy(L)
    #L = L.t()
    #adj = torch.mul(adj,L)
    #L.detach()
    adj = adj.cpu()
    #torch.cuda.empty_cache()
    z1 = F.normalize(z1)
    #print(adj.shape)
    batch_size = 1024
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    indices = torch.arange(0, num_nodes).to(device)
    losses = []
    #one_tensor = torch.ones(adj.shape[0],1)
    #one_tensor = one_tensor.to(device)
    #nei_count = dgl.sparse.spmm(adj, one_tensor) * 2 + 1
    nei_count = 0.5 * torch.sum(adj, 1)  + 1  # intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(torch.tensor(nei_count))
    nei_count = nei_count.to(device)
    f = lambda x: torch.exp(x / tau)
    #intra_view_sim = f(sim(z1, z1)).requires_grad_(True)
    #inter_view_sim = f(sim(z1, z2)).requires_grad_(True)
    #intra_view_sim = intra_view_sim.to(device)
    #inter_view_sim = inter_view_sim.to(device)
    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        intra_view_sim = f(sim(z1[mask], z1))
        intra_view_sim = intra_view_sim.to(device)
        #print(intra_view_sim.shape)
        aadj = adj[i * batch_size:(i + 1) * batch_size,:]
        aadj = aadj.to(device)
        #print((intra_view_sim.mul(aadj)).sum(1).shape)
        losses.append((intra_view_sim[:, i * batch_size:(i + 1) * batch_size].diag() + (intra_view_sim.mul(aadj)).sum(1)) / (intra_view_sim.sum(1)))
        #losses.append((intra_view_sim[:, i * batch_size:(i + 1) * batch_size].diag()) / (intra_view_sim.sum(1)))
        
    loss = torch.cat(losses)

    #intra_val = dgl.sparse.spmm(adj, intra_view_sim).sum(1)
    #inter_val = dgl.sparse.spmm(adj, inter_view_sim).sum(1)

    #loss = inter_view_sim.diag() / (intra_view_sim.sum(1) + inter_view_sim.sum(1))
    
    
    
    #loss = (intra_view_sim.diag()  + (intra_view_sim.mul(adj)).sum(1)) / (
            #intra_view_sim.sum(1))
            
            
            
    #loss = (inter_view_sim.diag() + dgl.sparse.spmm(adj, intra_view_sim).sum(0) + dgl.sparse.spmm(adj, inter_view_sim).sum(0)) / (
            #intra_view_sim.sum(1) + inter_view_sim.sum(1))
    loss = loss / nei_count  

    return -torch.log(loss)





