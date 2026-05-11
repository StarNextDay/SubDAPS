import torch
import numpy as np

class Deblurring():
    def __init__(self, kernel, channels, img_dim, device):
        self.img_dim   = img_dim
        self.channels  = channels
        _nextpow2 = lambda x : int(np.power(2, np.ceil(np.log2(x))))
        self.fft2_size = _nextpow2(img_dim + kernel.shape[1] - 1) # next pow 2
        self.kernel_size = (kernel.shape[-2], kernel.shape[-1])
        self.kernel = kernel
        self.init_kernel = kernel.detach().clone()
        self.update_kernel(kernel)
        self.device = device
        self.out_img_dim = img_dim
    
    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        ret = self.U(singulars * temp[:, :singulars.shape[1]])
        return ret

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[1]]))
    
    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        threshold = 0.05
        inv_singulars = torch.where(singulars > threshold, 1.0 / singulars, torch.ones_like(singulars))
        temp[:, :singulars.shape[1]] = temp[:, :singulars.shape[1]] * inv_singulars
        return self.V(self.add_zeros(temp))
    
    def Ht_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        threshold = 0.05
        inv_singulars = torch.where(singulars > threshold, 1.0 / singulars, torch.ones_like(singulars))
        temp[:, :singulars.shape[1]] = temp[:, :singulars.shape[1]] * inv_singulars
        return self.U(self.add_zeros(temp))
    

    def Sigma_pinv(self, vec):
        temp = vec.reshape(vec.shape[0], -1)
        singulars = self.singulars()
        threshold = 0.05
        inv_singulars = torch.where(singulars > threshold, 1.0 / singulars, torch.ones_like(singulars))
        temp[:, :singulars.shape[1]] = temp[:, :singulars.shape[1]] * inv_singulars
        return temp
    
    def Sigma(self, vec):
        temp = vec.reshape(vec.shape[0], -1)
        singulars = self.singulars()
        return self.add_zeros(singulars * temp[:, :singulars.shape[1]])

    def V(self, vec):
        vec = vec.reshape(vec.shape[0], self.channels, -1)
        vec = vec / self._singular_phases[:, None, :]
        vec = vec.reshape(vec.shape[0], self.channels, -1)
        vec = self._batch_inv_perm(vec, self._perm)
        vec_ifft = torch.fft.ifft2(vec.reshape(vec.shape[0], self.channels, self.fft2_size, self.fft2_size),\
            norm="ortho").real
        out = vec_ifft[:, :, :self.img_dim, :self.img_dim].reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        vec_fft = torch.fft.fft2(vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim), (self.fft2_size, self.fft2_size), norm="ortho")
        vec_fft = self._batch_perm(vec_fft.reshape(vec.shape[0], self.channels, -1), self._perm)
        vec_fft = vec_fft * self._singular_phases[:, None, :]
        return vec_fft.reshape(vec.shape[0], -1)

    def U(self, vec):
        vec = vec.reshape(vec.shape[0], self.channels, -1)
        vec = self._batch_inv_perm(vec, self._perm)
        vec_ifft = torch.fft.ifft2(vec.reshape(vec.shape[0], self.channels, self.fft2_size, self.fft2_size),\
            norm="ortho").real
        out = vec_ifft[:, :, self.kernel_size[0]//2:int(self.kernel_size[0]//2+self.img_dim), \
            self.kernel_size[1]//2:int(self.kernel_size[1]//2+self.img_dim)]
        
        return out

    def Ut(self, vec):
        _ks0 = self.kernel_size[0]
        _ks1 = self.kernel_size[1]
        _Nf  =  self.fft2_size
        exec_zeropad = torch.nn.ZeroPad2d((_ks0//2, _Nf-_ks0//2-self.img_dim,\
            _ks1//2, _Nf-_ks1//2-self.img_dim))
        vec = exec_zeropad(vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim))
        vec_fft = torch.fft.fft2(vec, (self.fft2_size, self.fft2_size), norm="ortho")
        vec_fft = self._batch_perm(vec_fft.reshape(vec.shape[0], self.channels, -1), self._perm)
        return vec_fft.reshape(vec.shape[0], -1)

    def singulars(self):
        bsz = self._singulars.shape[0]
        return self._singulars.repeat(1, 3).reshape(bsz, -1)

    def add_zeros(self, vec):
        tmp = torch.zeros(vec.shape[0], self.channels, self.fft2_size**2, device=vec.device, dtype=vec.dtype)
        reshaped = vec.clone().reshape(vec.shape[0], self.channels, -1)
        tmp[:, :, :reshaped.shape[2]] = reshaped
        return tmp.reshape(vec.shape[0], -1)
    
    def update_kernel(self, kernel):
        """
        Update the internal kernel and associated variables using the provided kernel tensor.

        Args:
            kernel (torch.Tensor): The kernel tensor for the update. It should have the same shape of self.kernel

        Returns:
            None
        """
        self.kernel = kernel
        self.k_fft = torch.fft.fft2(kernel, (self.fft2_size, self.fft2_size), norm="ortho")
        bsz = kernel.shape[0]
        _eps_singulars = 1e-10 * torch.randn_like(self.k_fft)
        self._singular_phases = ((self.k_fft + _eps_singulars) / torch.abs(self.k_fft + _eps_singulars)).reshape(bsz, -1)
        self._singulars = torch.abs(self.k_fft * self.fft2_size).reshape(bsz, -1)
        ZERO = 0.05
        self._singulars[self._singulars < ZERO] = 0.0
        self._singulars, self._perm = self._singulars.sort(descending=True)
        self._singular_phases = self._batch_perm(self._singular_phases.reshape(bsz, -1), self._perm)
    
    def _batch_perm(self, tensor, perm):
        bsz = tensor.shape[0]
        for i_bsz in range(bsz):
            if tensor.dim() == 2:
                tensor[i_bsz, :] = tensor[i_bsz, perm[i_bsz]]
            elif tensor.dim() == 3:
                tensor[i_bsz, :, :] = tensor[i_bsz, :, perm[i_bsz]]
        return tensor

    def _batch_inv_perm(self, tensor, perm):
        bsz = tensor.shape[0]
        for i_bsz in range(bsz):
            if tensor.dim() == 2:
                tensor[i_bsz, perm[i_bsz]] = tensor[i_bsz, :].clone()
            elif tensor.dim() == 3:
                tensor[i_bsz, :, perm[i_bsz]] = tensor[i_bsz, :, :].clone()
        return tensor

    def reset_Hupdate(self):
        self.update_kernel(self.init_kernel.detach().clone())

    def H_fftconv(self, x, kernel):
        x_fft = torch.fft.fft2(x.reshape(x.shape[0], self.channels, self.img_dim, self.img_dim), (self.fft2_size, self.fft2_size), norm="ortho")
        k_fft = torch.fft.fft2(kernel, (self.fft2_size, self.fft2_size), norm="ortho")[:, None, :, :]
        y_fft = k_fft * x_fft
        y_fftconv = torch.fft.ifft2(y_fft, norm="ortho").real * self.fft2_size
        shifts = (self.kernel_size[0]//2, self.kernel_size[1]//2)
        y_fftconv_clip = y_fftconv[:, :, shifts[0]:int(shifts[0]+self.img_dim), \
            shifts[1]:int(shifts[1]+self.img_dim)]
        return y_fftconv_clip

    def interp_y_0(self, y_0, x_0, sigma_0):
        x_fft = torch.fft.fft2(x_0.reshape(x_0.shape[0], self.channels, self.img_dim, self.img_dim), (self.fft2_size, self.fft2_size), norm="ortho")
        k_fft = torch.fft.fft2(self.kernel, (self.fft2_size, self.fft2_size), norm="ortho")[:, None, :, :]
        y_fft = k_fft * x_fft
        y_fftconv = torch.fft.ifft2(y_fft, norm="ortho").real * self.fft2_size
        shifts = (self.kernel_size[0]//2, self.kernel_size[1]//2)
        y_fftconv += sigma_0 * torch.randn_like(y_fftconv)
        y_fftconv[:, :, shifts[0]:int(shifts[0]+self.img_dim), \
                shifts[1]:int(shifts[1]+self.img_dim)] = y_0
        return y_fftconv
