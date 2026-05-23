import torch
import torch.nn as nn


def TV_LOSS(x):
    #print(x)
    b,c,h,w = x.shape
    grad_x = x[:,:,1:,:]-x[:,:,:-1,:]
    grad_y = x[:,:,:,1:]-x[:,:,:,:-1]
    tv = (grad_x.abs().sum()+grad_y.abs().sum())/(b*c*h*w)
    return tv


