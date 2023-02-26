## Visually Imbalanced Stereo Matching

<p>
  <img src="resources/good.jpg" width="50%" /> 
  <img src="resources/bad.jpg" width="50%" />
</p>

<div class="row">
  <div class="column">
    <<img src="resources/good.jpg" width="100%" /> 
  </div>
  <div class="column">
    <img src="resources/bad.jpg" width="100%" />
  </div>
</div>

This repository contains code for the paper: [Visually Imbalanced Stereo Matching](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Visually_Imbalanced_Stereo_Matching_CVPR_2020_paper.pdf).

If you find the work useful please consider citing our paper
```bibtex
@InProceedings{VISM_Liu_2020,
  title = {Visually Imbalanced Stereo Matching},
  author = {Liu, Yicun and Ren, Jimmy and Zhang, Jiawei and Liu, Jianbo and Lin, Mude},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
year = {2020}}
```

## Installation 
The code is built on torch-0.4.0 and torchvision-0.2.0. The code also requires compiling a customized CUDA kernel which is based on cuda-8.0. We encourage you to create a virtual environment with Virtualenv or Anaconda

Once you have cloned the repository, cd to the root directory and  👇

1. Install dependencies.
   ```Shell
   pip install -r requirements.txt 
   ```
   
2. Compile the customized CUDA kernels. The Dynamic Displacement Filter described in the paper is implemented with our own customized kernel. You'll be needing PyTorch 0.4.0 and cuda-8.0.
   ```Shell
   bash install.sh
   ```
   
3. Download pre-trained models of view restoration network from [[GoogleDrive]](https://drive.google.com/drive/folders/1plB6jOGFXLyVkDgQhLZZSR7lX494Rb6Z?usp=sharing) and unzip them under the `Visually-Imabalanced-Stereo` folder. You should expect `KPN.pth` in the root directory.  The models serve as pre-trained models for dynamic filters and pre-trained models to compute the preceptual loss.

4. Optional. Prepare the stereo matching network following instruction at [CRL](https://github.com/Artifineuro/crl). The output of View Synthesis Network will be used for Stereo Matching

## Inferencing a single image
This repository contains a toy KITTI dataset. To run inference for a single pair stereo images. 👇
```
bash run.sh
```
You should expect average PSNR and average SSIM as your output, as well as saved output images.

## Reproducing KITTI Results

To produce the experiment results in the paper. You'll need to prepare KITTI Stereo 2015 dataset. You'll need to download the prepared dataset from [GoogleDrive](https://drive.google.com/file/d/1qSb6VflBR66xseCI8JaMFEssL2XN1Hx_/view?usp=sharing), or from the official Kitti website. You should perserve your folder sturcture like this:

```
dataset\
   |--data_scene_flow
         |---test
         |---train
```

To reproduce the inference of view synthesis network. You'll need to modify `--scale_factor` from 1 to 5, 10, 20 to see the performance under different imbalance factor.
```
python train.py 
         --only_test 1 
         --input_nc 3 
         --dataset data_scene_flow 
         --gpu_num 1 
         --loss Smooth-L1 
         --batchSize 1 
         --testBatchSize 1 
         --loading_weights 1 
         --scale_factor 2 \
         --filter_size_horizontal 201 
         --filter_size_vertical 0  
         --image_width 1242 
         --image_height 375 
         --weight_source ours
```
### Trouble Shooting
We acknowledge that the code is slightly old. Error could easily arise from using the compiled CUDA kernel. We provide a troubleshooting to solve the error that you could face in [Here](resources/trouble_shooting.txt)

