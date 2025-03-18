import torch
from einops import rearrange, reduce, repeat
from scipy.special import iv

def compute_heat_kernel_batch(eigenvalue, eigenvector, t):
    hk_threshold = 1e-5
    num_samples = eigenvalue.shape[0]

    eigval = eigenvalue.type(torch.float) # b, n
    eigvec = eigenvector.type(torch.float) # b, n, n

    eigval = torch.exp(-1 * eigval) # b, n
    eigval = torch.mul(torch.ones_like(eigvec), eigval.unsqueeze(dim=1)) # b, n, n
    eigval = eigval ** t.reshape(-1, 1)

    left = torch.mul(eigvec, eigval)
    right = torch.transpose(eigvec, 1, 2)

    """hk = Uk^2(s\Lambda)U^T """
    hk = torch.matmul(left, right) # b, n, n
    hk[hk < hk_threshold] = 0

    hk_grad = torch.matmul(torch.matmul(left, -torch.diag_embed(eigenvalue.float())), right)
    hk_one = torch.ones_like(hk) 
    hk_zero = torch.zeros_like(hk) 
    hk_sign = torch.where(hk >= hk_threshold, hk_one, hk_zero)  
    hk_grad = torch.mul(hk_grad, hk_sign)

    return hk, hk_grad


def compute_heat_kernel(eigenvalue, eigenvector, t):
    hk_threshold = 1e-5
    hk_list = [] # Heat kernel
    hk_grad_list = [] # Gradient of heat kernel

    num_samples = eigenvalue.shape[0]

    for i in range(num_samples):
        tmp = eigenvalue[i].type(torch.float) # (# of ROIs)
        one_tmp = torch.ones_like(eigenvector[i]) # (# of ROIs, # of ROIs)
        eigval = torch.mul(one_tmp, torch.exp(-tmp).reshape(1, -1)) ** t.reshape(-1, 1) # (# of ROIs, # of ROIs)
        eigvec = eigenvector[i].type(torch.float) # (# of ROIs, # of ROIs)
        left = torch.mul(eigvec, eigval)
        right = eigvec.T
        hk = torch.matmul(left, right) # Compute heat kernel (# of ROIs, # of ROIs)
        hk[hk < hk_threshold] = 0
        
        hk_grad = torch.matmul(torch.matmul(left, -torch.diag(tmp)), right) # Compute gradient of heat kernel (# of ROIs, # of ROIs)
        hk_one = torch.ones_like(hk) 
        hk_zero = torch.zeros_like(hk) 
        hk_sign = torch.where(hk >= hk_threshold, hk_one, hk_zero)  
        hk_grad = torch.mul(hk_grad, hk_sign)

        hk_list.append(hk)
        hk_grad_list.append(hk_grad)
        
    hk_list = torch.stack(hk_list)
    hk_grad_list = torch.stack(hk_grad_list)
    
    return hk_list, hk_grad_list