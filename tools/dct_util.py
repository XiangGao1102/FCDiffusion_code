import numpy as np
import torch


def dct(x, norm=None):
    '''
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    '''
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V


def idct(X, norm=None):
    '''
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    '''
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    return x.view(*x_shape)


def dct_2d(x, norm=None):
    '''
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT_II of the signal over the last 2 dimensions
    '''
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    '''
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimension
    '''
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    '''
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT_II of the signal over the last 3 dimensions
    '''
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    '''
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter 'norm', see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimension
    '''
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


def low_pass(dct, threshold):
    '''
    dct: tensor of shape [... h, w]
    threshold: integer number above which to zero out
    '''
    h, w = dct.shape[-2], dct.shape[-1]
    assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
    vertical = torch.range(0, h-1)[..., None].repeat(1, w).cuda()
    horizontal = torch.range(0, w-1)[None, ...].repeat(h, 1).cuda()
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask > threshold, torch.zeros_like(dct), dct)
    return dct


def low_pass_and_shuffle(dct, threshold):
    '''
    dct: tensor of shape [... h, w]
    threshold: integer number above which to zero out
    '''
    h, w = dct.shape[-2], dct.shape[-1]
    assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
    vertical = torch.range(0, h-1)[..., None].repeat(1, w).cuda()
    horizontal = torch.range(0, w-1)[None, ...].repeat(h, 1).cuda()
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask > threshold, torch.zeros_like(dct), dct)
    for i in range(0, threshold + 1):         # 0 ~ threshold
        dct = shuffle_one_frequency_level(i, dct)
    return dct


def shuffle_one_frequency_level(n, dct_tensor):
    h_num = torch.arange(n + 1)
    h_num = h_num[torch.randperm(n + 1)]
    v_num = n - h_num
    dct_tensor_copy = dct_tensor.clone()
    for i in range(n + 1):  # 0 ~ n
        dct_tensor[:, :, i, n - i] = dct_tensor_copy[:, :, v_num[i], h_num[i]]
    return dct_tensor


def high_pass(dct, threshold):
    '''
    dct: tensor of shape [... h, w]
    threshold: integer number below which to zero out
    '''
    h, w = dct.shape[-2], dct.shape[-1]
    assert 0 <= threshold <= h + w - 2, 'invalid value of threshold'
    vertical = torch.range(0, h-1)[..., None].repeat(1, w).cuda()
    horizontal = torch.range(0, w-1)[None, ...].repeat(h, 1).cuda()
    mask = vertical + horizontal
    while len(mask.shape) != len(dct.shape):
        mask = mask[None, ...]
    dct = torch.where(mask < threshold, torch.zeros_like(dct), dct)
    return dct






