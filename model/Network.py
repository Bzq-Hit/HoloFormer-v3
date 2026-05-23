import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, to_2tuple, DropPath

import math

from einops import repeat

"""HoloFormer v3""" 

#########################################
########### window operation#############
def window_reverse(windows, win_size, H, W, dilation_rate=1): 
    # B ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

######## Embedding for q,k,v ########    
class ComplexLinearProjection(nn.Module): 
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim = dim
        self.inner_dim = inner_dim

        self.to_q_real = nn.Linear(dim, inner_dim, bias=bias)
        self.to_q_imag = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv_real = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.to_kv_imag = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape

        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)

        x_real, x_imag = x.real, x.imag
        kv_real, kv_imag = attn_kv.real, attn_kv.imag

        q_real_real = self.to_q_real(x_real)
        q_imag_imag = self.to_q_imag(x_imag)
        q_real_imag = self.to_q_real(x_imag)
        q_imag_real = self.to_q_imag(x_real)
        q_real = q_real_real - q_imag_imag
        q_imag = q_real_imag + q_imag_real
        q = torch.complex(q_real, q_imag)
        q = q.reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]

        kv_real_real = self.to_kv_real(kv_real)
        kv_imag_imag = self.to_kv_imag(kv_imag)
        kv_real_imag = self.to_kv_real(kv_imag)
        kv_imag_real = self.to_kv_imag(kv_real)
        kv_real_part = kv_real_real - kv_imag_imag
        kv_imag_part = kv_real_imag + kv_imag_real
        kv = torch.complex(kv_real_part, kv_imag_part)

        kv = kv.reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        return q, k, v # (B_, heads, N, dim_head)

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = 2 * (q_L * self.dim * self.inner_dim + kv_L * self.dim * self.inner_dim * 2)
        return flops

##########################################
###########complex window-based self-attention#############
class ComplexWindowAttention_v2(nn.Module): 
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., sm_var = 'real'):

        super().__init__()
        self.sm_var = sm_var
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.win_size[0])
        coords_w = torch.arange(self.win_size[1])
        coords = torch.meshgrid(coords_h, coords_w, indexing='xy')
        coords = torch.stack(coords)

        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.win_size[0] - 1
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = ComplexLinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj_real = nn.Linear(dim, dim)
        self.proj_imag = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, x, attn_kv=None, mask=None):
        x_real = x[..., 0]
        x_imaginary = x[..., 1]
        B_, N, C = x_real.shape

        x_complex = torch.complex(x_real, x_imaginary)  # (B_, N, C)

        q, k, v = self.qkv(x_complex, attn_kv)

        q = q * self.scale

        attn_logits = q @ k.transpose(-2, -1).conj()

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn_logits.real.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)

        
        attn_real = attn_logits.real + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn_real).type(torch.complex64)

        attn = self.attn_drop(attn)
  
        out = attn @ v 

        out = out.permute(0, 2, 1, 3).contiguous().view(B_, N, C)

        out_real = out.real
        out_imag = out.imag

        proj_real = self.proj_real(out_real) - self.proj_imag(out_imag)
        proj_imag = self.proj_real(out_imag) + self.proj_imag(out_real)

        proj_real = self.proj_drop(proj_real)
        proj_imag = self.proj_drop(proj_imag)
        x = torch.stack((proj_real, proj_imag), dim=-1) # B_ N C 2
        return x

# Complex Con2d:
class ComplexConv2d(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.imaginary_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x_real, x_imaginary):
        real_output = self.real_conv(x_real) - self.imaginary_conv(x_imaginary)
        imaginary_output = self.real_conv(x_imaginary) + self.imaginary_conv(x_real)
        return real_output, imaginary_output

# Complex ConvTranspose2d 
class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.real_deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.imaginary_deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x_real, x_imaginary):
        real_output = self.real_deconv(x_real) - self.imaginary_deconv(x_imaginary)
        imaginary_output = self.real_deconv(x_imaginary) + self.imaginary_deconv(x_real)
        return real_output, imaginary_output

# Complex input projection
class ComplexInputProj_v2(nn.Module): 
    def __init__(self, in_channel=1, out_channel=64, kernel_size=3, stride=1, norm_layer=None, leaky_slope=0.2):
        super().__init__()

        self.proj = ComplexConv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)

        self.leaky_relu_real = nn.LeakyReLU(leaky_slope, inplace=True)
        self.leaky_relu_imaginary = nn.LeakyReLU(leaky_slope, inplace=True)

        if norm_layer is not None:
            self.norm = norm_layer(out_channel * 2)
        else:
            self.norm = None
        
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x_real, x_imaginary):
        real_output, imaginary_output = self.proj(x_real, x_imaginary)
        real_output = self.leaky_relu_real(real_output)
        imaginary_output = self.leaky_relu_imaginary(imaginary_output)
        complex_output = torch.stack((real_output, imaginary_output), dim=-1) # B C H W 2(Real and Imaginary)

        complex_output = complex_output.permute(0, 2, 3, 1, 4) # B H W C 2

        complex_output = complex_output.flatten(1, 2).contiguous() # B H*W C 2

        if self.norm is not None:
            complex_output = self.norm(complex_output)

        return complex_output # B H*W C 2

# Complex Output Projection
class ComplexOutputProj(nn.Module): 
    def __init__(self, in_channel=64, out_channel=1, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = ComplexConv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2) 

        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x): # x: B H*W C 2
        x_real = x[..., 0]
        x_imaginary = x[..., 1]
        B, L, C = x_real.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x_real = x_real.transpose(1, 2).contiguous().view(B, C, H, W) 
        x_imaginary = x_imaginary.transpose(1, 2).contiguous().view(B, C, H, W)
        real_output, imaginary_output = self.proj(x_real, x_imaginary) # real & imag: B 1 H W 

        out = torch.cat((real_output, imaginary_output), dim=1) # B 2 H W
        return out

# Complex Downsample Block
class ComplexDownsample_v2(nn.Module): 
    def __init__(self, in_channel, out_channel):
        super(ComplexDownsample_v2, self).__init__()
        self.conv = ComplexConv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x): # B H*W C 2
        x_real = x[..., 0]
        x_imaginary = x[..., 1]
        B, L, C = x_real.shape
        H = int(math.sqrt(L))   
        W = int(math.sqrt(L))
        x_real = x_real.transpose(1, 2).contiguous().view(B, C, H, W)
        x_imaginary = x_imaginary.transpose(1, 2).contiguous().view(B, C, H, W)
        real_output, imaginary_output = self.conv(x_real, x_imaginary)
        real_output = real_output.flatten(2).transpose(1,2).contiguous() # B H*W C
        imaginary_output = imaginary_output.flatten(2).transpose(1,2).contiguous() # B H*W C
        out = torch.stack((real_output, imaginary_output), dim=-1) # B H*W C 2
        return out

# Complex Upsample Block
class ComplexUpsample_v2(nn.Module): 
    def __init__(self, in_channel, out_channel):
        super(ComplexUpsample_v2, self).__init__()
        self.deconv = ComplexConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x_real = x[..., 0]
        x_imaginary = x[..., 1]
        B, L, C = x_real.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x_real = x_real.transpose(1, 2).contiguous().view(B, C, H, W)
        x_imaginary = x_imaginary.transpose(1, 2).contiguous().view(B, C, H, W)
        real_output, imaginary_output = self.deconv(x_real, x_imaginary)
        real_output = real_output.flatten(2).transpose(1,2).contiguous() # B H*W C
        imaginary_output = imaginary_output.flatten(2).transpose(1,2).contiguous() # B H*W C
        out = torch.stack((real_output, imaginary_output), dim=-1) # B H*W C 2
        return out

# ComplexLayerNorm
class ComplexLayerNorm(nn.Module): 
    def __init__(self, dim):
        super().__init__()
        self.norm_real = nn.LayerNorm(dim)
        self.norm_imaginary = nn.LayerNorm(dim)

    def forward(self, x):
        real = self.norm_real(x[..., 0])
        imaginary = self.norm_imaginary(x[..., 1])
        return torch.stack((real, imaginary), dim=-1)

#########################################
###########complex feed-forward network #############
class ComplexMlp(nn.Module): 
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.LeakyReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_real = nn.Linear(in_features, hidden_features)
        self.fc1_imaginary = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2_real = nn.Linear(hidden_features, out_features)
        self.fc2_imaginary = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x_real = x[..., 0]
        x_imaginary = x[..., 1]
        x_mlp_1_real = self.fc1_real(x_real)-self.fc1_imaginary(x_imaginary)
        x_mlp_1_imaginary = self.fc1_real(x_imaginary)+self.fc1_imaginary(x_real)
        x_mlp_1_real = self.act(x_mlp_1_real)
        x_mlp_1_imaginary = self.act(x_mlp_1_imaginary)
        x_mlp_1_real = self.drop(x_mlp_1_real)
        x_mlp_1_imaginary = self.drop(x_mlp_1_imaginary)
        x_mlp_2_real = self.fc2_real(x_mlp_1_real)-self.fc2_imaginary(x_mlp_1_imaginary)
        x_mlp_2_imaginary = self.fc2_real(x_mlp_1_imaginary)+self.fc2_imaginary(x_mlp_1_real)
        x_mlp_2_real = self.drop(x_mlp_2_real)
        x_mlp_2_imaginary = self.drop(x_mlp_2_imaginary)
        x = torch.stack((x_mlp_2_real, x_mlp_2_imaginary), dim=-1)
        return x

#########################################
####### Complex LeWinTransformer ########
class ComplexLeWinTransformerBlock(nn.Module): 
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.LeakyReLU, norm_layer=ComplexLayerNorm,token_projection='linear',token_mlp='mlp', sm_var='real'):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim) # Complex LN
        self.attn = ComplexWindowAttention_v2( # Complex Window Attention
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection, sm_var= sm_var)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) # Complex LN
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn','mlp']:
            self.mlp = ComplexMlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) # Complex MLP
        else:
            raise Exception("FFN error!") 

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x, mask=None):
        x_real = x[..., 0]
        x_imaginary = x[..., 1]
        B, L, C = x_real.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1)
            input_mask_windows = window_partition(input_mask, self.win_size) # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
        
        shortcut_real = x_real
        shortcut_imaginary = x_imaginary
        x = self.norm1(x) # B L(H*W) C 2
        x_real = x[..., 0]
        x_imaginary = x[..., 1]
        x_real = x_real.view(B, H, W, C)
        x_imaginary = x_imaginary.view(B, H, W, C)
        
        # cyclic shift
        if self.shift_size > 0:
            shifted_x_real = torch.roll(x_real, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x_imaginary = torch.roll(x_imaginary, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x_real = x_real
            shifted_x_imaginary = x_imaginary

        # partition windows
        x_windows_real = window_partition(shifted_x_real, self.win_size) # nW*B, win_size, win_size, C  N*C->C
        x_windows_imaginary = window_partition(shifted_x_imaginary, self.win_size) # nW*B, win_size, win_size, C  N*C->C
        x_windows_real = x_windows_real.view(-1, self.win_size * self.win_size, C) # nW*B, win_size*win_size, C
        x_windows_imaginary = x_windows_imaginary.view(-1, self.win_size * self.win_size, C) # nW*B, win_size*win_size, C

        wmsa_in = torch.stack((x_windows_real, x_windows_imaginary), dim=-1) # nW*B, win_size*win_size, C, 2

        # W-MSA/SW-MSA
        attn_windows = self.attn(wmsa_in, mask=attn_mask)  # nW*B, win_size*win_size, C, 2

        # merge windows
        attn_windows_real = attn_windows[..., 0] # nW*B, win_size*win_size, C
        attn_windows_imaginary = attn_windows[..., 1] # nW*B, win_size*win_size, C
        attn_windows_real = attn_windows_real.view(-1, self.win_size, self.win_size, C) # nW*B, win_size, win_size, C
        attn_windows_imaginary = attn_windows_imaginary.view(-1, self.win_size, self.win_size, C) # nW*B, win_size, win_size, C
        shifted_x_real = window_reverse(attn_windows_real, self.win_size, H, W) # B H W C
        shifted_x_imaginary = window_reverse(attn_windows_imaginary, self.win_size, H, W) # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x_real = torch.roll(shifted_x_real, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x_imaginary = torch.roll(shifted_x_imaginary, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_real = shifted_x_real
            x_imaginary = shifted_x_imaginary

        x_real = x_real.view(B, H * W, C) # B H*W C
        x_imaginary = x_imaginary.view(B, H * W, C)

        # FFN
        x_real = shortcut_real + self.drop_path(x_real)
        x_imaginary = shortcut_imaginary + self.drop_path(x_imaginary)
        x = torch.stack((x_real, x_imaginary), dim=-1) # B H*W C 2

        shortcut_ffn = x
        x = self.norm2(x) # B H*W C 2
        x = self.mlp(x) # B H*W C 2
        x = torch.stack((self.drop_path(x[..., 0]), self.drop_path(x[..., 1])), dim=-1) 
        x = shortcut_ffn + x # B H*W C 2

        del attn_mask
        return x


#########################################
########### Complex Basic layer of Uformer ################
class ComplexBasicUformerLayer(nn.Module): 
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=ComplexLayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='mlp', shift_flag=True, sm_var='real'):
        
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                ComplexLeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0 if (i % 2 == 0) else win_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    act_layer=nn.LeakyReLU,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp, sm_var=sm_var)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                ComplexLeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    shift_size=0,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    act_layer=nn.LeakyReLU,
                                    norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp, sm_var=sm_var)
            for i in range(depth)])
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
    
    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x,mask)
        return x


class ComplexUformer(nn.Module): 
    def __init__(self, img_size=256, in_chans=1, dd_in=1, 
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=ComplexLayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='mlp',
                 dowsample=ComplexDownsample_v2, upsample=ComplexUpsample_v2, shift_flag=True, sm_var = 'real'
                 , **kwargs):
        super().__init__()

        self.dtype = torch.float32
        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in
        self.in_chans = in_chans
        
        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))] 
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # Input/Output
        self.input_proj = ComplexInputProj_v2(in_channel=dd_in, out_channel=embed_dim, kernel_size=3, stride=1)
        self.output_proj = ComplexOutputProj(in_channel=2*embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = ComplexBasicUformerLayer(dim=embed_dim,
                            output_dim=embed_dim,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[0],
                            num_heads=num_heads[0],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            sm_var = sm_var)
        
        self.dowsample_0 = dowsample(embed_dim, embed_dim*2)

        self.encoderlayer_1 = ComplexBasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[1],
                            num_heads=num_heads[1],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            sm_var = sm_var)
        
        self.dowsample_1 = dowsample(embed_dim*2, embed_dim*4)

        self.encoderlayer_2 = ComplexBasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            sm_var = sm_var)
        
        self.dowsample_2 = dowsample(embed_dim*4, embed_dim*8)

        self.encoderlayer_3 = ComplexBasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            sm_var = sm_var)
        
        self.dowsample_3 = dowsample(embed_dim*8, embed_dim*16)

        # Bottleneck
        self.conv = ComplexBasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 4),
                                                img_size // (2 ** 4)),
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=conv_dpr,
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            sm_var = sm_var)
        
        # Decoder
        self.upsample_0 = upsample(embed_dim*16, embed_dim*8)
        self.decoderlayer_0 = ComplexBasicUformerLayer(dim=embed_dim*16,
                            output_dim=embed_dim*16,
                            input_resolution=(img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size, 
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[5]],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            sm_var = sm_var)
        
        self.upsample_1 = upsample(embed_dim*16, embed_dim*4)
        self.decoderlayer_1 = ComplexBasicUformerLayer(dim=embed_dim*8,
                            output_dim=embed_dim*8,
                            input_resolution=(img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            sm_var = sm_var)
        
        self.upsample_2 = upsample(embed_dim*8, embed_dim*2)
        self.decoderlayer_2 = ComplexBasicUformerLayer(dim=embed_dim*4,
                            output_dim=embed_dim*4,
                            input_resolution=(img_size // 2,
                                                img_size // 2),
                            depth=depths[7],
                            num_heads=num_heads[7],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate,attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            sm_var = sm_var)
        
        self.upsample_3 = upsample(embed_dim*4, embed_dim)
        self.decoderlayer_3 = ComplexBasicUformerLayer(dim=embed_dim*2,
                            output_dim=embed_dim*2,
                            input_resolution=(img_size,
                                                img_size),
                            depth=depths[8],
                            num_heads=num_heads[8],
                            win_size=win_size,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                            norm_layer=norm_layer,
                            use_checkpoint=use_checkpoint,
                            token_projection=token_projection,token_mlp=token_mlp,shift_flag=shift_flag,
                            sm_var = sm_var)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"
    
    def forward(self, x, mask=None):

        # divide real and imag
        x_real = x[:,0,:,:].unsqueeze(1) # B 1 H W
        x_imag = x[:,1,:,:].unsqueeze(1) # B 1 H W

        y = self.input_proj(x_real, x_imag)
            
        y = self.pos_drop(y)

        # Encoder
        conv0 = self.encoderlayer_0(y,mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0,mask=mask)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1,mask=mask)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2,mask=mask)
        pool3 = self.dowsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3,mask=mask)

        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat((up0, conv3), dim=2) 
        deconv0 = self.decoderlayer_0(deconv0,mask=mask)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat((up1, conv2), dim=2)
        deconv1 = self.decoderlayer_1(deconv1,mask=mask)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat((up2, conv1), dim=2)
        deconv2 = self.decoderlayer_2(deconv2,mask=mask)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat((up3, conv0), dim=2)
        deconv3 = self.decoderlayer_3(deconv3,mask=mask)

        # Output Projection
        y = self.output_proj(deconv3) # B 2 H W
        return x + y



