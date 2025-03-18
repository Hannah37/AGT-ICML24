import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt

def kl_divergence(p, q):
    EPSILON = 1e-8
    p = torch.clamp(p, min=EPSILON) 
    q = torch.clamp(q, min=EPSILON) 
    return torch.sum(p * torch.log(p / q))

def euclidean_distance(X1, X2):
    return torch.norm(X1 - X2, p='fro')

def jensenshannon(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

### Layer 2
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self): # Initialize weights and bias
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj): # Graph convolution
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_dim, hid_dim) 
        self.gc2 = GraphConvolution(hid_dim, out_dim) 
        self.dropout = dropout
        self.f = None
        self.rdp = None
        self.rdp2 = None
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, adj): # Graph convolution part 
        ### Graph convolution 1
        tmp = self.gc1.forward(x, adj)
        x2 = F.relu(tmp)

        with torch.no_grad():
            x_one = torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.where(tmp <= 0., x_zero, x_one)
            tmp = x2

        x2 = F.dropout(x2, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.mul(self.rdp, torch.where((tmp != 0.) & (x2 == 0.), x_zero, x_one))

        self.f = x2
        
        ### Graph convolution 2
        x3 = self.gc2.forward(x2, adj)

        if self.training:
            self.final_conv_acts = x3
            self.final_conv_acts.register_hook(self.activations_hook)

        x4 = F.relu(x3)

        with torch.no_grad():
            x_one = torch.ones_like(x3)
            x_zero = torch.zeros_like(x3)
            self.rdp2 = torch.where(x4 <= 0., x_zero, x_one)

        return x4

class AGT(nn.Module):
    def __init__(self, args, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(AGT, self).__init__()
        self.args = args
        self.dropout = dropout

        '''stack(Xs and X)'''
        self.gcn = GCN(in_dim, hid_dim, hid_dim, dropout) # Graph convolution
        self.linear = nn.Linear(adj_dim * hid_dim, adj_dim * hid_dim // 2) # Readout 1
        self.linear2 = nn.Linear(adj_dim * hid_dim // 2, out_dim) # Readout 2
        
        self.device = args.device

        '''trainable scales!!! n_scales = n_class x n_ROI'''
        self.t = torch.nn.Parameter(torch.FloatTensor(args.nclass, adj_dim).uniform_(args.t_init_min, args.t_init_max), requires_grad=True) # adj_dim == n_ROI 

        '''non-adaptive scales used for ablation study '''
        # self.t = torch.ones(args.nclass, adj_dim).cuda(args.device) * -0.5

    def get_scales(self):
        return self.t

    ''' parallel & per-class scale'''
    def classwise_graphspace_heat_kernel(self, eigenvalue, eigenvector, t, label, is_train):
        hk_threshold = 1e-5
        n_batch = eigenvalue.shape[0]
        n_roi = eigenvalue.shape[1]

        eigval = eigenvalue.type(torch.float) # b, n
        eigvec = eigenvector.type(torch.float) # b, n, n

        eigval = torch.exp(-2 * eigval) # b, n
        eigval = torch.mul(torch.ones_like(eigvec), eigval.unsqueeze(dim=1)) # b, n, n

        batch_indices = torch.arange(n_batch)
        eigval = eigval[batch_indices, :, :] ** t[label].view(-1, 1, n_roi) # b, n, n

        left = torch.mul(eigvec, eigval)
        right = torch.transpose(eigvec, 1, 2)

        """hk = Uk^2(s\Lambda)U^T """
        hk = torch.matmul(left, right) # b, n, n
        hk[hk < hk_threshold] = 0
        
        return hk
    
    """ stack(Xs and X)"""
    def forward(self, x, adj, eigenvalue, eigenvector, label, is_train=True):  
        '''classwise hk '''
        heat_kernel = self.classwise_graphspace_heat_kernel(eigenvalue, eigenvector, self.t, label, is_train)

        xs = torch.einsum('bnm, bmp-> bnp', heat_kernel, x) 
        x_combined = torch.cat([x, xs], dim=-1).squeeze()

        x_0 = self.gcn.forward(x_combined, adj) # (batch x n_ROI x hidden_dim) 
        x_0 = rearrange(x_0, 'b r d -> b (r d)')
        dist_loss = self.dist_between_cls(x_0, label)

        x1 = self.linear(x_0) # Readout 1
        x2 = F.relu(x1)
        x3 = self.linear2(x2) # Readout 2

        return F.log_softmax(x3, dim=1), dist_loss


    """
    x: (# subjects, #ROI x hiddedn_dim)
    """
    def dist_between_cls(self, x, label):
        num_classes = torch.unique(label).numel()

        if num_classes < 3:
            return 0
        else:
            class_prob_ls = [torch.mean(x[label == class_idx], dim=0) for class_idx in range(num_classes)]

            # shape: c-1, distance between adjacent classes
            dist_adj = [euclidean_distance(class_prob_ls[i], class_prob_ls[i+1]) for i in range(num_classes - 1)]

            # shape: c-2, distance between 2-hop adjacent classes
            dist_adj_2 = [euclidean_distance(class_prob_ls[i], class_prob_ls[i+2]) for i in range(num_classes - 2)]

            loss = F.relu(dist_adj[0] + dist_adj[1] - dist_adj_2[0])
            for class_idx in range(1, num_classes - 2):
                loss = loss + F.relu(dist_adj[class_idx] + dist_adj[class_idx + 1] - dist_adj_2[class_idx])

            return loss / (num_classes - 2)
