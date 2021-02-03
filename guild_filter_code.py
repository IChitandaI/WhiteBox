import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def box_filter(x, r):
    k_size=int(r*2+1)
    ch=x.shape[1]
    weight=1/(k_size**2)
    box_kernel=weight*torch.ones((ch, 1, k_size, k_size), dtype=x.dtype, device=x.device)
    return F.conv2d(x, box_kernel, padding=r, groups=ch)



def guide_filter(x, y, r, eps=1e-2):
    x_shape=x.shape
    N=box_filter(torch.ones((1, 1, x_shape[2], x_shape[3]), dtype=x.dtype, device=x.device), r)

    mean_x=box_filter(x, r)/N
    mean_y=box_filter(y, r)/N
    cov_xy=box_filter(x*y, r)/N-mean_x*mean_y
    var_x=box_filter(x*x, r)/N-mean_x*mean_x

    A=cov_xy/(var_x+eps)
    b=mean_y-A*mean_x
    mean_A=box_filter(A, r)/N
    mean_b=box_filter(b, r)/N

    output=mean_A*x+mean_b

    return output