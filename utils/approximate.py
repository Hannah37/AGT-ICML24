import torch
from einops import rearrange, reduce, repeat
from scipy.special import iv


#######################################################################################
####################################### Exact #########################################
############################ Exact heat kernel batch computation ############################
#######################################################################################
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