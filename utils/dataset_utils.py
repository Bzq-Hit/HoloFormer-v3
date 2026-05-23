import torch
from torch.nn.functional import pad

def generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance, pad_size=None, device='cpu'):
    """
    Generate the otf from [0,pi] not [-pi/2,pi/2] using torch
    :param wavelength:
    :param nx:
    :param ny:
    :param deltax:
    :param deltay:
    :param distance:
    :return:
    """
    if not isinstance(nx, torch.Tensor):
        nx = torch.tensor(nx)
    if not isinstance(ny, torch.Tensor):
        ny = torch.tensor(ny)
    if not isinstance(deltax, torch.Tensor):
        deltax = torch.tensor(deltax)
    if not isinstance(deltay, torch.Tensor):
        deltay = torch.tensor(deltay)
    if not isinstance(distance, torch.Tensor):
        distance = torch.tensor(distance)
    if not isinstance(wavelength, torch.Tensor):
        wavelength = torch.tensor(wavelength)

    if pad_size:
        nx = pad_size[0]
        ny = pad_size[1]
    nx = torch.tensor(nx).to(device)
    ny = torch.tensor(ny).to(device)
    r1 = torch.linspace(-nx / 2, nx / 2 - 1, nx).to(device)
    c1 = torch.linspace(-ny / 2, ny / 2 - 1, ny).to(device)
    deltaFx = 1 / (nx * deltax) * r1
    deltaFy = 1 / (nx * deltay) * c1
    mesh_qx, mesh_qy = torch.meshgrid(deltaFx, deltaFy)
    k = 2 * torch.pi / wavelength
    otf = torch.exp(
        1j * k.to(device) * distance.to(device) * torch.sqrt(1 - wavelength.to(device) ** 2 * (mesh_qx.to(device) ** 2
                                                                                               + mesh_qy.to(
                    device) ** 2)))
    otf = torch.fft.ifftshift(otf).to(device)
    return otf

def psnr(x, im_orig):
    def norm_tensor(x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x)+1e-8)

    x = norm_tensor(x)
    im_orig = norm_tensor(im_orig)
    mse = torch.mean(torch.square(im_orig - x))
    psnr = torch.tensor(10.0) * torch.log10(1 / mse)
    if psnr>1000:
        return torch.tensor(0.0)
    return psnr

class DH_operator:
    def __init__(self, **prop_kernel):
        self.A = generate_otf_torch(**prop_kernel)
        self.AT = torch.conj(self.A)
        self.device = 'cpu'
        self.nx = prop_kernel['nx']
        self.ny = prop_kernel['ny']
        self.pad_size = prop_kernel.get('pad_size', None)

    def forward(self, data, **kwargs):
        self.device = data.device
        self.normalized = kwargs.get('normalized', False)

        if self.pad_size:
            h_p = (self.pad_size[0] - data.shape[-2]) // 2
            w_p = (self.pad_size[1] - data.shape[-1]) // 2
            if len(data.shape) == 3:
                data = data.unsqueeze(0)
                data = pad(data, [h_p, h_p, w_p, w_p], mode='replicate')[0, :, :, :]

            elif len(data.shape) == 2:
                data =  data.unsqueeze(0).unsqueeze(0)
                data = pad(data, [h_p,h_p,w_p,w_p], mode='replicate')[0,0,:,:]

            elif len(data.shape) == 4:
                data = pad(data, [h_p, h_p, w_p, w_p], mode='replicate')
            else:
                raise Exception(f'Unsupported shape {data.shape}')

        fs_out = torch.multiply(torch.fft.fft2(data), self.A.expand(data.shape).to(self.device))
        f_out = torch.fft.ifft2(fs_out)
        ampitude = f_out.abs()
        phase = f_out.angle()
        return ampitude, phase

    def holo_backward(self, data):
        self.device = data.device
        data = torch.sqrt(data) 
        fs_out = torch.multiply(torch.fft.fft2(data), self.AT.expand(data.shape).to(self.device))
        f_out = torch.fft.ifft2(fs_out)
        amplitude = f_out.abs()
        phase = f_out.angle()
        return amplitude, phase



