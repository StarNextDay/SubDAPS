# Image Restoration via Diffusion Models with Dynamic Resolution (ICML 2026)

This is the official implementation of "Image Restoration via Diffusion Models with Dynamic Resolution
". This paper has been accpeted by [ICML 2026](https://icml.cc/). 

## Abstract
Diffusion Models (DMs) have exhibited remarkable efficacy in various image restoration tasks. However, existing approaches typically operate within the high-dimensional pixel space, resulting in high computational overhead. While methods based on latent DMs seek to alleviate this issue by utilizing the compressed latent space of a variational autoencoder (VAE), they require repeated encoder-decoder inference. This introduces significant additional computational burdens, often resulting in runtime performance that is even inferior to that of their pixel-space counterparts. To mitigate the computational inefficiency, this work proposes projecting data into lower-dimensional subspaces using dynamic resolution DMs to accelerate the inference process. We first fine-tune pre-trained DMs for dynamic resolution priors and adapt DPS and DAPS, which are two widely used pixel-space methods for general image restoration tasks, into the proposed framework, yielding methods we refer to as SubDPS and SubDAPS, respectively. Given the favorable inference speed and reconstruction fidelity of SubDAPS, we introduce an enhanced variant termed SubDAPS++ to further boost both reconstruction efficiency and quality. Empirical evaluations across diverse image datasets and various restoration tasks demonstrate that the proposed methods outperform recent DM-based approaches in the majority of experimental scenarios.

![title](images/pipeline.png)

## Getting started 

### (1) Clone the repository

```
git clone https://github.com/StarNextDay/SubDAPS.git

cd SubDAPS
```


### (2) Download pretrained checkpoint

From the [link](https://drive.google.com/drive/folders/1RMLqG_c7tWVzQlQrFA8bL0rRGQrEmPEh?usp=drive_link), download the checkpoint "ffhq_10m.pt" and paste it to ./models/;

From the [link](https://drive.google.com/drive/folders/1RMLqG_c7tWVzQlQrFA8bL0rRGQrEmPEh?usp=drive_link), download the checkpoint "imagenet256.pt" and paste it to ./models/;
```
mkdir models
mv {DOWNLOAD_DIR}/ffhq_10m.pt ./model/ffhq_10m.pt
mv {DOWNLOAD_DIR}/imagenet256.pt ./model/imagenet.pt
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.


### (3) Set environment

We use the external codes for motion-blurring and non-linear deblurring.

```
git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse

git clone https://github.com/LeviBorodenko/motionblur motionblur
```

From the [link](https://drive.google.com/file/d/1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy/view), download the checkpoint "GOPRO_wVAE.pth" and paste it to ./bkse/experiments/pretrained/.
```
mv {DOWNLOAD_DIR}/GOPRO_wVAE.pt ./bkse/experiments/pretrained/
```
{DOWNLOAD_DIR} is the directory that you downloaded checkpoint to.

Install dependencies

```
conda env create -f environment.yml
conda activate subspace
```

## Inference
```
### example for Motion deblurring task
#### DMILO
python pipeline.py --dataset "ffhq" --task "motion_deblur" -n 5 --cuda 0
```

## Citation

If you find our work interesting, please consider citing
```
@inproceedings{zheng2026image,
  title = {Image Restoration via Diffusion Models with Dynamic Resolution},
  author={Zheng, Yang and Li, Wen and Liu, Zhaoqiang},
  booktitle={ICML},
  year={2026}
}
```