import torch
import torch.nn as nn
from utils.model_utils import *
from utils.general import *
from utils.dataset_utils import *

import math

seed = 11
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class Pipe(nn.Module):
    def __init__(self, config, **kwargs):
        super(Pipe, self).__init__()
        self.model_W = create_model(config['MODEL'])
        
        self.num_params = network_parameters(self.model_W) 
        self.prop_kernel = config['MEASUREMENT']['prop_kernel']

        self.operator = DH_operator(**self.prop_kernel,device = 'cuda')
        self.amp_pred = None
        self.phase_pred = None
        self.device = 'cpu'

    def forward(self, x):
        self.device = x.device

        self.amp_in, self.phase_in = self.operator.holo_backward(x)
        
        x_complex = torch.polar(self.amp_in, self.phase_in)
        real = x_complex.real
        imag = x_complex.imag

        x_holo_2_real_and_imag = torch.cat([real, imag], dim=1)  # B,2,H,W

        out_W = self.model_W(x_holo_2_real_and_imag)  # (B,2,H,W)
        self.real_pred = out_W[:, 0:1, :, :]
        self.imag_pred = out_W[:, 1:2, :, :]

        self.complex_map = torch.complex(self.real_pred, self.imag_pred) 

        self.amp_pred = torch.abs(self.complex_map)
        self.phase_pred = torch.angle(self.complex_map)

        out_amp, out_phase = self.operator.forward(self.complex_map)
        out = torch.polar(out_amp, out_phase)
        out = out.abs()
        out = out ** 2

        return out

    def rescale_phase(self, phase, range=[-1, 1]):
        return (phase - phase.min()) / (phase.max() - phase.min()) * (range[1] - range[0]) + range[0]

    def update_kernel(self, depth_ratio, verbose=False):
        self.operator = DH_operator(**self.prop_kernel,device =self.device)
        return

    def summary(self):
        print("The number of parameters in model", network_parameters(self.model_W))
