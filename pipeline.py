import argparse, os, yaml
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from guided_diffusion.unet import *
from motionblur.motionblur import Kernel
import scipy
from guided_diffusion.script_util import *
import lpips
import glob
import json
from tqdm import tqdm

import os.path

import torch.nn.functional as F
import hdf5storage
from torch.func import jvp, functional_call
import yaml
import argparse

class NonlinearBlurOperator():
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)
        self.random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        self.random_kernel.requires_grad = False
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        for param in blur_model.parameters():
            param.requires_grad = False
        return blur_model
    
    def forward(self, data, **kwargs):
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=self.random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

class DownsampleOperator(nn.Module):
    """
    Downsampling Operator: 2x2 Average Pooling
    
    Logic: Transforms an HxW image to (H/2)x(W/2). Each new pixel represents 
    the average value of the 4 pixels in the corresponding 2x2 region of the original image.
    """
    def __init__(self):
        super().__init__()
        # kernel_size=2, stride=2 reduces both height and width by half
        self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)

class UpsampleOperator(nn.Module):
    """
    Upsampling Operator: 2x Scaling
    
    Logic: Restores an (H/2)x(W/2) image back to HxW.
    Uses 'nearest' neighbor interpolation, which is often used as a simple 
    structural inverse to the pooling operation.
    """
    def __init__(self):
        super().__init__()
        # scale_factor=2 doubles both height and width
        self.op = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.op(x)

def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= -1
    img /= 2
    img = np.clip(img, a_min=0, a_max=1)
    return img

def clear_color(x: torch.Tensor) -> np.ndarray:
    if torch.is_complex(x):
        x = torch.abs(x)
    if x.shape[1] == 3:
        x = x.detach().cpu().squeeze().numpy()
        return normalize_np(np.transpose(x, (1,2,0)))
    elif x.shape[1] == 1:
        x = x.detach().cpu().squeeze().numpy()
        return normalize_np(x)
    else:
        x = x.detach().cpu().squeeze().numpy()
        return x


### None Operator
class Identity:
    def forward(self, x):
        return x
### Choose the Closest Solution

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def batch(model, logdir, dataset='celeba', n=100, task=None, device='cuda', begin=0, end=-1, noise_level = 0.05):
    from utils.deblur import Deblurring
    from utils import utils_model
    from utils import utils_sisr as sr
    from utils import utils_image as util
    from utils.utils_resizer import Resizer
    from utils.utils_inpaint import mask_generator
    import shutil
    with torch.no_grad():
        ####### Preparation
        model = model.eval()
        dtype = torch.float32
        tasks = ['inpainting', 'box', 'super_resolution', 'gaussian_deblur', 'motion_deblur', 'nonlinear_deblur', 'hdr']
        assert task in tasks, f'Invalid task: {task}'
        image_paths = sorted(glob.glob(f'./data/{dataset}/*.png'))[:n]
        total_psnrs, total_ssims, total_lpipss = [], [], []
        log = os.path.join(logdir, 'log.txt')
        with open(log, "a") as f:
            print(f'SubDAPS++: {n} Pictures', file = f)
            print(f'Noise Level: {noise_level}', file = f)
        record = os.path.join(logdir, "record")
        os.makedirs(record, exist_ok=True)
        imgdir = os.path.join(logdir, "images")
        os.makedirs(imgdir, exist_ok=True)
        codedir = os.path.join(logdir, 'code')
        os.makedirs(codedir, exist_ok=True)
        code_names = ['pipeline.py']
        for code_name in code_names:
            shutil.copyfile(code_name, os.path.join(codedir, code_name))
        if end == -1:
            end = n
        image_paths = image_paths[begin:end]
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        ###
        
        ####### Task Configuration
        if task == 'inpainting' or task == 'box':
            step = 10
            iter_num = 100
            thres = 1e-5
        elif task == 'super_resolution':
            step = 10
            sf = 4
            iter_num = 100
            thres = 1e-5
        elif task == 'gaussian_deblur':
            step = 10
            sf = 1
            kernel_size = 61
            iter_num = 100
            thres = 1e-4
        elif task == 'motion_deblur':
            step = 10
            sf = 1
            kernel_size = 61
            iter_num = 100
            thres = 3e-4
        elif task == 'hdr':
            step = 10
            iter_num = 100
            thres = 1e-5
        elif task == 'nonlinear_deblur':
            step = 100
            iter_num = 100
            thres = 1e-4
        else:
            assert False
        ###
        
        ####### Hyperparameter
        beta_start = 0.0001
        beta_end = 0.02
        num_train_timesteps = 1000
        betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        betas                   = torch.from_numpy(betas).to(device)
        alphas                  = 1.0 - betas
        alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
        sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
        reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image    
        t_start = num_train_timesteps - 1
        diffusion_steps = 1000
        learn_sigma = True
        noise_schedule = 'linear'
        use_kl = False
        predict_xstart = False
        rescale_timesteps = False
        rescale_learned_sigmas = False
        timestep_respacing = ""
        diffusion = create_gaussian_diffusion(
            steps=diffusion_steps,
            learn_sigma=learn_sigma,
            noise_schedule=noise_schedule,
            use_kl=use_kl,
            predict_xstart=predict_xstart,
            rescale_timesteps=rescale_timesteps,
            rescale_learned_sigmas=rescale_learned_sigmas,
            timestep_respacing=timestep_respacing,
        )
        sigma = noise_level
        sigmas = []
        sigma_ks = []
        lambdas = []
        rhos = []
        for i in range(num_train_timesteps):
            sigmas.append(reduced_alpha_cumprod[num_train_timesteps-1-i])
            sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
            lambdas.append(torch.log(sqrt_alphas_cumprod[num_train_timesteps-1-i] / sqrt_1m_alphas_cumprod[num_train_timesteps-1-i]))
            rhos.append((sigma**2)/(sigma_ks[i]**2))
        sigma = noise_level / 2
        rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(sigma_ks).to(device)
        lambdas = torch.tensor(lambdas).to(device)
        ###
        lambda_seq = torch.linspace(torch.min(lambdas), torch.max(lambdas), iter_num)
        ### EDM
        rho_edm = 7
        sigma_k_max_inv = torch.max(sigma_ks) ** (1 / rho_edm)
        sigma_k_min_inv = torch.min(sigma_ks) ** (1 / rho_edm)
        ramp = torch.linspace(0, 1, iter_num).to(device)
        sigma_ks_seq = (sigma_k_max_inv + ramp * (sigma_k_min_inv - sigma_k_max_inv)) ** rho_edm
        ###### Upsample (64 * 64 -> 128 * 128 -> 256 * 256)
        upsample_time_0 = int(iter_num / 3 * 1) # 64 to 128
        upsample_time_1 = int(iter_num / 3 * 2) # 128 to 256
    ########################################################################## Iteration
    for img_path in tqdm(image_paths, desc="Processing Images", unit="image"):
        with torch.no_grad():
            ####### Dataloader
            seed = int(os.path.splitext(os.path.basename(img_path))[0])
            torch_seed(seed)
            max_psnr = None
            max_ssim = None
            min_lpips = None
            model.eval()
            dtype = torch.float32
            gt_img_path = img_path
            gt_img = Image.open(gt_img_path).convert("RGB")
            ref_numpy = np.array(gt_img) / 255.0
            ref_x = ref_numpy * 2 - 1
            ref_x = ref_x.transpose(2, 0, 1)
            ref_img = torch.Tensor(ref_x).to(dtype).to(device).unsqueeze(0)
            ref_img.requires_grad = False
            y = (ref_img + 1) / 2 # (1, 3, 256,256) [0, 1]
            y.requires_grad = False
            ###
            ####### Operator
            if task == 'inpainting':
                mask_gen = mask_generator(mask_type='random', mask_prob_range=[0.7, 0.7])
                mask = mask_gen._retrieve_random(ref_img) 
                y_n = y * mask + sigma * torch.randn_like(y)
                def degrade_op(x):
                    x_blurs = x * mask
                    return x_blurs     
                x = y_n
            elif task == 'box':
                mask = torch.ones_like(y)
                h, w = y.shape[2], y.shape[3]
                box_size = h // 2
                top = (h - box_size) // 2
                left = (w - box_size) // 2
                mask[:, :, top:top + box_size, left:left + box_size] = 0
                y_n = y * mask + sigma * torch.randn_like(y)
                def degrade_op(x):
                    x_blurs = x * mask
                    return x_blurs   
                x = y_n
            elif task =='super_resolution':
                degrade_op = Resizer((1, 3, 256, 256), 1 / sf).to(device)
                y_n = degrade_op(y)
                y_n = y_n + sigma * torch.randn_like(y_n)
                x = F.interpolate(y_n, size=(y_n.shape[2]*sf, y_n.shape[2]*sf), mode='bicubic', align_corners=False).to(device)
                kernels = hdf5storage.loadmat(os.path.join('', 'kernels', 'kernels_bicubicx234.mat'))['kernels']
                k_index = sf - 2 if sf < 5 else 2
                k = kernels[0, k_index].astype(np.float64)
                k = util.single2tensor4(np.expand_dims(k, 2)).to(device) 
            elif task == 'gaussian_deblur':
                std = 3.0
                n = np.zeros((kernel_size, kernel_size))
                n[kernel_size // 2, kernel_size // 2] = 1
                k = scipy.ndimage.gaussian_filter(n, sigma=std)
                k = torch.from_numpy(k).type(torch.float32)
                k = k.to(device).view(1, kernel_size, kernel_size)
                deblur = Deblurring(k, 3, 256, device)                
                def degrade_op(x):
                    x_blurs = deblur.H(x)
                    return x_blurs                
                y_n = degrade_op((y * 2 - 1)) / 2 + 0.5 + sigma * torch.randn_like(y)
                x = y_n
            elif task =='motion_deblur':
                std = 0.5
                k = Kernel(size=(kernel_size, kernel_size), intensity=0.5).kernelMatrix
                k = torch.from_numpy(k).type(torch.float32)
                k = k.to(device).view(1, kernel_size, kernel_size)
                deblur = Deblurring(k, 3, 256, device)                
                def degrade_op(x):
                    x_blurs = deblur.H(x)
                    return x_blurs                
                y_n = degrade_op((y * 2 - 1)) / 2 + 0.5 + sigma * torch.randn_like(y)
                x = y_n
            elif task == 'nonlinear_deblur':
                opt_yml_path = './bkse/options/generate_blur/default.yml'
                operator = NonlinearBlurOperator(opt_yml_path=opt_yml_path, device=device)
                def degrade_op(x):
                    result = operator.forward(x)
                    return result
                y_n = degrade_op((y * 2 - 1)) / 2 + 0.5 + sigma * torch.randn_like(y)
                x = y_n
            elif task == 'hdr':
                scale = 2
                def degrade_op(x):
                    return torch.clip((x * scale), -1, 1)
                y_n = degrade_op((y * 2 - 1)) / 2 + 0.5 + sigma * torch.randn_like(y)
                x = y_n
            y = y_n
            y_n = y_n * 2 - 1
            ###        
            ####### Save 
            picture_name = os.path.basename(img_path)
            tmp_imgdir = os.path.join(imgdir, os.path.splitext(picture_name)[0])
            os.makedirs(tmp_imgdir, exist_ok=True)
            plt.imsave(os.path.join(tmp_imgdir, 'measurement.png'), clear_color(y_n.clone()))
            plt.imsave(os.path.join(tmp_imgdir, 'origin.png'), clear_color(ref_img.clone()))
            ###
            up_op = UpsampleOperator().to(device)
            down_op = DownsampleOperator().to(device)
            x = sqrt_alphas_cumprod[t_start] * (x * 2 - 1)
            x = down_op(down_op(x))
            x = x + sqrt_1m_alphas_cumprod[t_start] * torch.randn_like(x)
        ####### Main Loop
        model_out_type = 'pred_xstart'
        x0s = []
        mid_i = 0
        Convergence = False
        temp_diff = 0
        x_c = x.clone()
        for i in tqdm(range(len(sigma_ks_seq))):
            # time step associated with the log SNR \lambda
            curr_sigma_k = sigma_ks_seq[i]
            t_i = (torch.abs(sigma_ks - curr_sigma_k)).argmin()
            idx = num_train_timesteps - 1 - t_i
            curr_sigma = sigmas[idx].cpu().numpy()
            # skip iters
            if t_i > t_start:
                continue
            # repeat for semantic consistence: from repaint
            # --------------------------------
            # step 1, reverse diffsuion step
            # --------------------------------
            with torch.no_grad():
                x0 = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                            model_diffusion=model, diffusion=diffusion, ddim_sample=False, alphas_cumprod=alphas_cumprod)
            # -------------------------------- 
            # step 2, Conjuagete Gradient Step
            # --------------------------------
            if sigma_ks_seq[i] != sigma_ks_seq[-1]:
                x0hat = x0.clone().requires_grad_(True)
                # first order solver
                measurement = (y * 2 - 1)
                x0hat.grad = None
                if i <= upsample_time_0:
                    temp_x0hat = up_op(up_op(x0hat))
                elif i <= upsample_time_1:
                    temp_x0hat = up_op(x0hat)
                else:
                    temp_x0hat = x0hat
                ax0hat = degrade_op(temp_x0hat) 
                difference = torch.sum((measurement - ax0hat) ** 2) + rhos[t_i] * torch.sum((x0hat - x0) ** 2)
                difference = difference / 2
                grad = torch.autograd.grad(outputs=difference, inputs=x0hat)[0]
                with torch.no_grad():
                    g = -grad
                    d = g.clone()
                for _ in range(step):
                    if task != 'hdr':
                        def forward_operator(x):
                            if i <= upsample_time_0:
                                res = up_op(up_op(x))
                            elif i <= upsample_time_1:
                                res = up_op(x)
                            else:
                                res = x
                            return degrade_op(res)
                    else:
                        def forward_operator(x):
                            if i <= upsample_time_0:
                                res = up_op(up_op(x))
                            elif i <= upsample_time_1:
                                res = up_op(x)
                            else:
                                res = x
                            scale = 2 
                            val = res * scale
                            abs_val = torch.abs(val)
                            tail = 1 + torch.tanh(abs_val - 1)
                            out = torch.where(abs_val <= 1, val, torch.sign(val) * tail)
                            return out
                    ######
                    with torch.no_grad():
                        _, omega = jvp(forward_operator, (x0hat,), (d,))
                        alpha = (torch.sum(g * d)) / (rhos[t_i] * torch.sum(d * d) + torch.sum(omega * omega))
                        x0hat = x0hat + alpha * d
                    ######
                    x0hat.requires_grad_(True)
                    x0hat.grad = None
                    if i <= upsample_time_0:
                        temp_x0hat = up_op(up_op(x0hat))
                    elif i <= upsample_time_1:
                        temp_x0hat = up_op(x0hat)
                    else:
                        temp_x0hat = x0hat
                    ax0hat = degrade_op(temp_x0hat)
                    difference = torch.sum((measurement - ax0hat) ** 2) + rhos[t_i] * torch.sum((x0hat - x0) ** 2)
                    difference = difference / 2
                    grad = torch.autograd.grad(outputs=difference, inputs=x0hat)[0]
                    g_next = -grad
                    d = g_next + (torch.sum(g_next * g_next) / torch.sum(g * g)) * d
                    g = g_next
                with torch.no_grad():
                    temp_diff = torch.mean((x0 - x0hat) ** 2)
                x0 = x0hat.detach_()
            x0s.append(x0)
            # --------------------------------
            # add noise back to t_{i-1}
            # --------------------------------
            
            if not (sigma_ks_seq[i] == sigma_ks_seq[-1]):
                t_im1 = (torch.abs(sigma_ks - sigma_ks_seq[i+1])).argmin()
                
                # calculate \hat{\eposilon}
                if i == upsample_time_0 or i == upsample_time_1:
                    x0 = up_op(x0)
                if i <= upsample_time_1 or temp_diff >= thres:
                    x = sqrt_alphas_cumprod[t_im1] * x0 +  sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x0)
                else:
                    if not Convergence:
                        Convergence = True
                        mid_i = i
                        x_c = x.clone()
                    noise = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                    x = sqrt_alphas_cumprod[t_im1] * x0 +  sqrt_1m_alphas_cumprod[t_im1] * noise
        result = x.detach().clone()
        
        ###########################################################################################################
        for i in range(mid_i, len(sigma_ks_seq)):
            # time step associated with the log SNR \lambda
            curr_sigma_k = sigma_ks_seq[i]
            t_i = (torch.abs(sigma_ks - curr_sigma_k)).argmin()
            # skip iters
            if t_i > t_start:
                continue
            if not (sigma_ks_seq[i] == sigma_ks_seq[-1]):
                # with torch.no_grad():
                x0_c = x0s[i]
                x0_prev = x0s[i+1]
                alpha_i = sqrt_alphas_cumprod[t_i]
                sigma_i = sqrt_1m_alphas_cumprod[t_i]
                lambda_i = torch.log(1 / sigma_ks_seq[i])
                # lambda_i = torch.log(alpha_i / sigma_i)
                t_im1 = (torch.abs(sigma_ks - sigma_ks_seq[i+1])).argmin()
                alpha_im1 = sqrt_alphas_cumprod[t_im1]
                sigma_im1 = sqrt_1m_alphas_cumprod[t_im1]
                lambda_im1 = torch.log(1 / sigma_ks_seq[i+1])
                # lambda_im1 = torch.log(alpha_i / sigma_i)
                h = lambda_im1 - lambda_i
                if h == 0:
                    continue
                I1 = sigma_im1 * (torch.exp(lambda_im1) - torch.exp(lambda_i))
                # I2 = alpha_im1 - (alpha_im1 - sigma_im1 * alpha_i / sigma_i) / h
                I2 = (sigma_im1 * alpha_i / sigma_i) - I1 / h
                ########
                if i == upsample_time_0 or i == upsample_time_1:
                    x_c = up_op(x_c)
                    x_c = sigma_im1 / sigma_i * x_c + I1 * x0_prev + I2 * (x0_prev - up_op(x0_c))
                else:
                    x_c = sigma_im1 / sigma_i * x_c + I1 * x0_prev + I2 * (x0_prev - x0_c)
        result = x_c.detach().clone()
        ###########################################################################################################
        ####### Evaluation
        with torch.no_grad():
            output = torch.clamp(result.detach().clone(), -1, 1)
            output_numpy = output.detach().cpu().squeeze().numpy()
            output_numpy = (output_numpy + 1) / 2
            output_numpy = np.transpose(output_numpy, (1, 2, 0))
            # calculate psnr
            tmp_psnr = peak_signal_noise_ratio(ref_numpy, output_numpy)
            # calculate ssim
            tmp_ssim = structural_similarity(ref_numpy, output_numpy, channel_axis=2, data_range=1)
            # calculate lpips
            rec_img_torch = torch.from_numpy(output_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
            gt_img_torch = torch.from_numpy(ref_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
            rec_img_torch = rec_img_torch * 2 - 1
            gt_img_torch = gt_img_torch * 2 - 1
            lpips_alex = loss_fn_alex(gt_img_torch, rec_img_torch).item()
            if max_psnr is None:
                max_psnr = tmp_psnr
                plt.imsave(os.path.join(tmp_imgdir, 'result.png'), clear_color(output.clone()))
            if max_ssim is None:
                max_ssim = tmp_ssim
            if min_lpips is None:
                min_lpips = lpips_alex
            if tmp_psnr > max_psnr:
                max_psnr = tmp_psnr
                max_ssim = tmp_ssim
                min_lpips = lpips_alex
                plt.imsave(os.path.join(tmp_imgdir, 'result.png'), clear_color(output.clone()))
        with open(log, "a") as f:
            picture_name = os.path.basename(img_path)
            print(f'{picture_name}: psnr: {max_psnr}, ssims: {max_ssim}, lpips: {min_lpips}', file = f)
        ### Save record as Json
        record_dir = os.path.join(record, os.path.splitext(picture_name)[0] + '.json')
        with open(record_dir, "w") as f:
            json.dump({
                "psnr": max_psnr,
                "ssim": max_ssim,
                "lpips_alex": min_lpips
            }, f)
        total_psnrs.append(max_psnr)
        total_ssims.append(max_ssim)
        total_lpipss.append(min_lpips)
        ###
        
    # Calculate avg and std
    avg_psnr = np.mean(total_psnrs)
    avg_ssim = np.mean(total_ssims)
    avg_lpips = np.mean(total_lpipss)
    with open(log, "a") as f:
        picture_name = os.path.basename(img_path)
        print(f'avg psnr: {avg_psnr}, avg ssims: {avg_ssim}, avg lpips: {avg_lpips}', file = f)
    std_psnr = np.std(total_psnrs)
    std_ssim = np.std(total_ssims)
    std_lpips = np.std(total_lpipss)
    with open(log, "a") as f:
        picture_name = os.path.basename(img_path)
        print(f'std psnr: {std_psnr}, std ssims: {std_ssim}, std lpips: {std_lpips}', file = f)
    return avg_psnr, avg_ssim, avg_lpips


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="logdir",
        default="./SubDAPS++"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="dataset",
        default="ffhq"
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        nargs="?",
        help="number of test images",
        default=10
    )
    parser.add_argument(
        "-bn",
        "--begin_number",
        type=int,
        nargs="?",
        help="begin number",
        default=0
    )
    parser.add_argument(
        "-en",
        "--end_number",
        type=int,
        nargs="?",
        help="end number",
        default=-1
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        nargs="?",
        default='inpainting'
    )
    parser.add_argument(
        "--cuda",
        type=int,
        nargs="?",
        help="cuda device ID",
        default=1
    )
    
    parser.add_argument(
        "--noise_level",
        type=float,
        nargs="?",
        help="Noise Level",
        default=0.05
    )
    return parser

def torch_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    torch_seed(123) ### Set the random Seed
    # Load configurations
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    img_model_config = 'model_configs/model_config_{}.yaml'.format(opt.dataset)
    device = torch.device(f"cuda:{opt.cuda}" if torch.cuda.is_available() else "cpu")
    img_model_config = load_yaml(img_model_config)
    model = create_model_(**img_model_config)
    model = model.to(device)
    model.eval()
    import time
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # Put timeStep into the Log directory
    logdir = os.path.join(opt.logdir, opt.dataset, opt.task, timestamp)
    os.makedirs(logdir,exist_ok=True)
    # Intermediate Layer Optimization
    batch(model, logdir, dataset=opt.dataset, n=opt.number, task=opt.task, device=device, begin=opt.begin_number, end=opt.end_number, noise_level=opt.noise_level)