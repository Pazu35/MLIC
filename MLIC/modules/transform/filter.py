import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF

class LearnableFIRLowpass(nn.Module):
    def __init__(self, numtaps=101, init_cutoff=0.107):
        super().__init__()
        try:
            from scipy.signal import firwin

            init_kernel = firwin(numtaps, init_cutoff)
        except Exception:
            # Fallback if SciPy is unavailable: Hamming-windowed sinc low-pass.
            n = torch.arange(numtaps, dtype=torch.float32)
            m = (numtaps - 1) / 2.0
            fc = float(init_cutoff)
            x = n - m
            h = 2.0 * fc * torch.sinc(2.0 * fc * x)
            w = torch.hamming_window(numtaps, periodic=False, dtype=torch.float32)
            init_kernel = (h * w)
            init_kernel = (init_kernel / init_kernel.sum()).cpu().numpy()

        self.kernel = nn.Parameter(torch.tensor(init_kernel, dtype=torch.float32).view(1, 1, -1))
        self.pad = numtaps // 2


    def forward(self, x):
        x = x.unsqueeze(1)
        y = F.conv1d(x, self.kernel.to(dtype=x.dtype, device=x.device), padding=self.pad)
        return y.squeeze(1)




class LearnableDoGSmoother(nn.Module):
    def __init__(self, init_sigma1=1.0, init_sigma2=4.0, radius=15):
        super().__init__()
        self.raw_sigma1 = nn.Parameter(torch.tensor(init_sigma1).log())
        self.raw_sigma2 = nn.Parameter(torch.tensor(init_sigma2).log())
        self.mix = nn.Parameter(torch.tensor(0.5))  # sigmoid-squashed
        self.radius = radius

    def _gauss(self, sigma, device, dtype):
        pos = torch.arange(-self.radius, self.radius + 1, dtype=dtype, device=device)
        k = torch.exp(-0.5 * (pos / sigma) ** 2)
        return k / k.sum()

    def forward(self, x):
        x = x.unsqueeze(1)
        s1, s2 = self.raw_sigma1.exp(), self.raw_sigma2.exp()
        w = torch.sigmoid(self.mix)
        kernel = (w * self._gauss(s1, x.device, x.dtype) +
                  (1 - w) * self._gauss(s2, x.device, x.dtype)).view(1, 1, -1)
        x_padded = F.pad(x, (self.radius, self.radius), mode='replicate')
        return F.conv1d(x_padded, kernel, padding=0)



class LearnableGaussianSmoother(nn.Module):
    def __init__(self, init_sigma=2.0, radius=15):
        super().__init__()
        self.raw_sigma = nn.Parameter(torch.tensor(init_sigma).log())  # log-param -> always positive
        self.radius = radius  # fixed kernel support, tune to ~3x expected sigma

    def forward(self, x):
        x = x.unsqueeze(1)
        sigma = self.raw_sigma.exp()
        pos = torch.arange(-self.radius, self.radius + 1, dtype=x.dtype, device=x.device)
        kernel = torch.exp(-0.5 * (pos / sigma) ** 2)
        kernel = (kernel / kernel.sum()).view(1, 1, -1)

        x_padded = F.pad(x, (self.radius, self.radius), mode='replicate')
        return F.conv1d(x_padded, kernel, padding=0)


def reflect_pad_1d(x, pad_len):
    # x: (..., L) along last dim
    left = x[..., 1:pad_len+1].flip(-1)   # mirror, excluding the edge sample itself
    right = x[..., -pad_len-1:-1].flip(-1)
    return torch.cat([left, x, right], dim=-1)

def zero_phase_lfilter(x, a_coeffs, b_coeffs, pad_len=30):
    # x: (..., L) along last dim
    # reflect-pad to absorb the transient before/after the real signal
    x_padded = reflect_pad_1d(x, pad_len)

    y = AF.lfilter(x_padded, a_coeffs, b_coeffs, clamp=False, batching=True)
    y = y.flip(-1)
    y = AF.lfilter(y, a_coeffs, b_coeffs, clamp=False, batching=True)
    y = y.flip(-1)

    return y[..., pad_len:-pad_len]  # trim back to original length