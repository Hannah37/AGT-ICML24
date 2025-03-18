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

### Layer 1
class GraphConvolution1(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution1, self).__init__()
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

class GCN1(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution1(in_dim, out_dim) 
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
            
        self.f = x
        
        return x2

class GLAS1(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(GLAS1, self).__init__()
        self.dropout = dropout
        self.gcn = GCN1(in_dim, hid_dim, hid_dim, dropout) # Graph convolution
        self.linear = nn.Linear(adj_dim * hid_dim, adj_dim * hid_dim // 2) # Readout 1
        self.linear2 = nn.Linear(adj_dim * hid_dim // 2, out_dim) # Readout 2
        self.linrdp = None
        self.linrdp2 = None

    """
    x: (# subjects, # ROI features, # used features)
    adj: (# subjects, # ROI features, # ROI features)
    """
    def forward(self, x, adj): 
        xs = x.shape[0] # (# subjects)
        x = self.gcn.forward(x, adj).reshape(xs, -1) # Graph convolution

        x1 = self.linear(x) # Readout 1
        x2 = F.relu(x1)

        with torch.no_grad():
            x_one = torch.ones_like(x1)
            x_zero = torch.zeros_like(x1)
            self.linrdp = torch.where(x2 <= 0., x_zero, x_one)

        x3 = self.linear2(x2) # Readout 2
        x4 = x3
        
        with torch.no_grad():
            x_one = torch.ones_like(x3)
            x_zero = torch.zeros_like(x3)
            self.linrdp2 = torch.where(x4 <= 0., x_zero, x_one)
        
        return F.log_softmax(x4, dim=1)


### Layer 2
class GraphConvolution2(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution2, self).__init__()
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

        # print('gcnweight.grad: ', self.weight.grad)
        # print('gcn weight: ', self.weight[0][0], self.weight[0][1])

        return output

class GCN2(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN2, self).__init__()
        self.gc1 = GraphConvolution2(in_dim, hid_dim) 
        self.gc2 = GraphConvolution2(hid_dim, out_dim) 
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

class GLAS2(nn.Module):
    def __init__(self, args, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(GLAS2, self).__init__()
        self.args = args
        self.dropout = dropout

        '''stack(Xs and X)'''
        self.gcn = GCN2(in_dim, hid_dim, hid_dim, dropout) # Graph convolution
        self.linear = nn.Linear(adj_dim * hid_dim, adj_dim * hid_dim // 2) # Readout 1
        self.linear2 = nn.Linear(adj_dim * hid_dim // 2, out_dim) # Readout 2
        
        self.device = args.device

        '''n_scales = n_class x n_ROI'''
        # self.t = torch.nn.Parameter(torch.FloatTensor(args.nclass, adj_dim).uniform_(args.t_init_min, args.t_init_max), requires_grad=True) # adj_dim == n_ROI 

        '''non-adaptive scales'''
        self.t = torch.ones(args.nclass, adj_dim).cuda(args.device) * -0.5
        print(self.t)

    def get_scales(self):
        return self.t

    ''' parallel '''
    def graphspace_heat_kernel(self, eigenvalue, eigenvector, t):
        hk_threshold = 1e-5

        eigval = eigenvalue.type(torch.float) # b, n
        eigvec = eigenvector.type(torch.float) # b, n, n

        eigval = torch.exp(-2 * eigval) # b, n
        eigval = torch.mul(torch.ones_like(eigvec), eigval.unsqueeze(dim=1)) # b, n, n
        eigval = eigval ** t.reshape(-1, 1)

        left = torch.mul(eigvec, eigval)
        right = torch.transpose(eigvec, 1, 2)

        """hk = Uk^2(s\Lambda)U^T """
        hk = torch.matmul(left, right) # b, n, n
        hk[hk < hk_threshold] = 0

        return hk
    
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
    
    def vizualize(self, args, eigval, label, path):
        # print(label, label.shape)
        # print(eigval, eigval.shape)
        
        device = args.device
        eigval = eigval.to(device)
        label = label.to(device)

        sum_eigval = torch.matmul(label.T, eigval) # c, n

        class_counts = label.sum(dim=0)
        avg_eigval = sum_eigval.T / class_counts.unsqueeze(0).clamp(min=1)
        avg_eigval = avg_eigval.T # c, n

        color = ['deepskyblue', 'orange', 'green', 'red', 'black']
        
        '''save exp(-2t\lambda)'''
        exp_t = torch.exp(-2 * torch.mul(self.t, avg_eigval))
        for i in range(5):
            ''' y-axis: exp(-2t\lambda), x-axis: eigval'''
            plt.scatter(avg_eigval[i].cpu().detach(), exp_t[i].cpu().detach(), c=color[i], label=f'Class {i}')     
            
            plt.xlabel('eigval')
            plt.ylabel('exp(-2t\lambda)')
            plt.ylim(torch.min(exp_t).item(), torch.max(exp_t).item())
            plt.legend()
            plt.title('Scatter Plots for Each Class')
            plt.savefig(os.path.join(path, 'exp(-2tv)_' + str(i) + '.png'))
            plt.close()

        for i in range(5):
            ''' y-axis: exp(-2t\lambda), x-axis: eigval'''
            plt.scatter(avg_eigval[i].cpu().detach(), exp_t[i].cpu().detach(), c=color[i], label=f'Class {i}')     
        plt.xlabel('eigval')
        plt.ylabel('exp(-2t\lambda)')
        # plt.ylim(torch.min(exp_t).item(), torch.max(exp_t).item())
        plt.legend()
        plt.title('Scatter Plots for Each Class')
        plt.savefig(os.path.join(path, 'exp(-2tv)_all-in-one.png'))
        plt.close()

        '''save t'''
        for i in range(5):
            ''' y-axis: t, x-axis: eigval'''
            plt.scatter(avg_eigval[i].cpu().detach(), self.t[i].cpu().detach(), c=color[i], label=f'Class {i}')      
            
            plt.xlabel('eigval')
            plt.ylabel('t')
            plt.ylim(torch.min(self.t).item(), torch.max(self.t).item())
            plt.legend()
            plt.title('Scatter Plots for Each Class')
            plt.savefig(os.path.join(path, 't_' + str(i) + '.png'))
            plt.close()

        for i in range(5):
            ''' y-axis: t, x-axis: eigval'''
            plt.scatter(avg_eigval[i].cpu().detach(), self.t[i].cpu().detach(), c=color[i], label=f'Class {i}') 
   
        plt.xlabel('eigval')
        plt.ylabel('t')
        # plt.ylim(torch.min(self.t).item(), torch.max(self.t).item())
        plt.legend()
        plt.title('Scatter Plots for Each Class')
        plt.savefig(os.path.join(path, 't_all-in-one.png'))
        plt.close()


    """ stack(Xs and X)"""
    def forward(self, x, adj, eigenvalue, eigenvector, label, is_train=True):  
        '''classwise hk '''
        heat_kernel = self.classwise_graphspace_heat_kernel(eigenvalue, eigenvector, self.t, label, is_train)

        xs = torch.einsum('bnm, bmp-> bnp', heat_kernel, x) # Xs (b, 160, 1)
        # x_combined = torch.stack([x, xs], dim=-1).squeeze() # adni
        x_combined = torch.cat([x, xs], dim=-1).squeeze()

        x_0 = self.gcn.forward(x_combined, adj) # (batch x n_ROI x hidden_dim) == (b, 160, 16)
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

    
    
### Layer 3
class GraphConvolution3(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution3, self).__init__()
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

class GCN3(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN3, self).__init__()
        self.gc1 = GraphConvolution3(in_dim, hid_dim) 
        self.gc2 = GraphConvolution3(hid_dim, hid_dim) 
        self.gc3 = GraphConvolution3(hid_dim, out_dim) 
        self.dropout = dropout
        self.f = None
        self.f2 = None
        self.rdp = None
        self.rdp2 = None
        self.rdp3 = None
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
            x3 = x4
            
        x4 = F.dropout(x4, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x4)
            x_zero = torch.zeros_like(x4)
            self.rdp = torch.mul(self.rdp2, torch.where((x3 != 0.) & (x4 == 0.), x_zero, x_one))

        self.f2 = x4
        
        ### Graph convolution 3
        x5 = self.gc3.forward(x4, adj)
        #x6 = F.relu(x5)
        x6 = x5

        if self.training:
            self.final_conv_acts = x5
            self.final_conv_acts.register_hook(self.activations_hook)

        with torch.no_grad():
            x_one = torch.ones_like(x6)
            x_zero = torch.zeros_like(x6)
            self.rdp3 = torch.where(x6 <= 0., x_zero, x_one)

        return x6

class GLAS3(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(GLAS3, self).__init__()
        self.dropout = dropout
        self.gcn = GCN3(in_dim, hid_dim, hid_dim, dropout) # Graph convolution
        self.linear = nn.Linear(adj_dim * hid_dim, adj_dim * hid_dim // 2) # Readout 1
        self.linear2 = nn.Linear(adj_dim * hid_dim // 2, out_dim) # Readout 2
        self.linrdp = None
        self.linrdp2 = None

    """
    x: (# subjects, # ROI features, # used features)
    adj: (# subjects, # ROI features, # ROI features)
    """
    def forward(self, x, adj): 
        xs = x.shape[0] # (# subjects)
        x = self.gcn.forward(x, adj).reshape(xs, -1) # Graph convolution

        x1 = self.linear(x) # Readout 1
        x2 = F.relu(x1)

        with torch.no_grad():
            x_one = torch.ones_like(x1)
            x_zero = torch.zeros_like(x1)
            self.linrdp = torch.where(x2 <= 0., x_zero, x_one)

        x3 = self.linear2(x2) # Readout 2
        x4 = x3
        
        with torch.no_grad():
            x_one = torch.ones_like(x3)
            x_zero = torch.zeros_like(x3)
            self.linrdp2 = torch.where(x4 <= 0., x_zero, x_one)
        
        return F.log_softmax(x4, dim=1) 
    

### 4 Layer
class GraphConvolution4(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution4, self).__init__()
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

class GCN4(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(GCN4, self).__init__()
        self.gc1 = GraphConvolution4(in_dim, hid_dim) 
        self.gc2 = GraphConvolution4(hid_dim, hid_dim) 
        self.gc3 = GraphConvolution4(hid_dim, hid_dim)
        self.gc4 = GraphConvolution4(hid_dim, out_dim)
        self.dropout = dropout
        self.f = None
        self.f2 = None
        self.f3 = None
        self.rdp = None
        self.rdp2 = None
        self.rdp3 = None
        self.rdp4 = None
        self.final_conv_acts = None
        self.final_conv_grads = None

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, adj): # Graph convolution part 
        ### Graph convolution 1
        x1 = self.gc1.forward(x, adj)
        x2 = F.relu(x1)

        with torch.no_grad():
            x_one = torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.where(x1 <= 0., x_zero, x_one)
            x1 = x2

        x2 = F.dropout(x2, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x2)
            x_zero = torch.zeros_like(x2)
            self.rdp = torch.mul(self.rdp, torch.where((x1 != 0.) & (x2 == 0.), x_zero, x_one))

        self.f = x2
        
        ### Graph convolution 2
        x3 = self.gc2.forward(x2, adj)
        x4 = F.relu(x3)

        if self.training:
            self.final_conv_acts = x3
            self.final_conv_acts.register_hook(self.activations_hook)
            
        with torch.no_grad():
            x_one = torch.ones_like(x4)
            x_zero = torch.zeros_like(x4)
            self.rdp2 = torch.where(x4 <= 0., x_zero, x_one)
            x3 = x4

        x4 = F.dropout(x4, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x4)
            x_zero = torch.zeros_like(x4)
            self.rdp = torch.mul(self.rdp2, torch.where((x3 != 0.) & (x4 == 0.), x_zero, x_one))

        self.f2 = x4
        
        ### Graph convolution 3
        x5 = self.gc3.forward(x4, adj)
        x6 = F.relu(x5)
        
        if self.training:
            self.final_conv_acts = x5
            self.final_conv_acts.register_hook(self.activations_hook)

        with torch.no_grad():
            x_one = torch.ones_like(x6)
            x_zero = torch.zeros_like(x6)
            self.rdp3 = torch.where(x6 <= 0., x_zero, x_one)
            x5 = x6
            
        x6 = F.dropout(x6, self.dropout, training=self.training)

        with torch.no_grad():
            x_one = (1. / (1. - self.dropout)) * torch.ones_like(x6)
            x_zero = torch.zeros_like(x6)
            self.rdp2 = torch.mul(self.rdp3, torch.where((x5 != 0.) & (x6 == 0.), x_zero, x_one))

        self.f3 = x6
        
        ### Graph convolution 4
        x7 = self.gc4.forward(x6, adj)
        x8 = x7
        
        if self.training:
            self.final_conv_acts = x7
            self.final_conv_acts.register_hook(self.activations_hook)

        with torch.no_grad():
            x_one = torch.ones_like(x8)
            x_zero = torch.zeros_like(x8)
            self.rdp4 = torch.where(x8 <= 0., x_zero, x_one)

        return x8

class GLAS4(nn.Module):
    def __init__(self, adj_dim, in_dim, hid_dim, out_dim, dropout):
        super(GLAS4, self).__init__()
        self.dropout = dropout
        self.gcn = GCN4(in_dim, hid_dim, hid_dim, dropout) # Graph convolution
        self.linear = nn.Linear(adj_dim * hid_dim, adj_dim * hid_dim // 2) # Readout 1
        self.linear2 = nn.Linear(adj_dim * hid_dim // 2, out_dim) # Readout 2
        self.linrdp = None
        self.linrdp2 = None

    """
    x: (# subjects, # ROI features, # used features)
    adj: (# subjects, # ROI features, # ROI features)
    """
    def forward(self, x, adj): 
        xs = x.shape[0] # (# subjects)
        x = self.gcn.forward(x, adj).reshape(xs, -1) # Graph convolution

        x1 = self.linear(x) # Readout 1
        x2 = F.relu(x1)

        with torch.no_grad():
            x_one = torch.ones_like(x1)
            x_zero = torch.zeros_like(x1)
            self.linrdp = torch.where(x2 <= 0., x_zero, x_one)

        x3 = self.linear2(x2) # Readout 2
        x4 = x3
        
        with torch.no_grad():
            x_one = torch.ones_like(x3)
            x_zero = torch.zeros_like(x3)
            self.linrdp2 = torch.where(x4 <= 0., x_zero, x_one)
        
        return F.log_softmax(x4, dim=1)