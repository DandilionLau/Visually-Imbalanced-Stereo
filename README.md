## Visually Imbalanced Stereo Matching

---

This repository contains code for the paper: [Visually Imbalanced Stereo Matching](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Visually_Imbalanced_Stereo_Matching_CVPR_2020_paper.pdf).


#### Prerequisites   
The code is originally built on torch-0.4.1 and torchvision-0.2.0. Running the code requires compiling a customized CUDA kernel, which is tested on CUDA-8.0.

#### Installation
1. Clone the code. 
   ```Shell  
   git clone https://github.com/DandilionLau/Visually-Imbalanced-Stereo 
   cd Visually-Imabalanced-Stereo
   ```

2. Install dependencies.
   ```Shell
   pip install -r requirements.txt 
   ```
   
3. Compile customized CUDA kernels. Due to lack of support to implement the dynamic filters under torch when the paper is written, we wrote our own customized kernel and complied based on CUDA-8.0 and pytorch-0.4.1 for the original code. 
   ```Shell
   bash install.sh
   ```
   
4. Download pre-trained models of view restoration network.   
   Download pre-trained models from [[GoogleDrive]](https://drive.google.com/drive/folders/1plB6jOGFXLyVkDgQhLZZSR7lX494Rb6Z?usp=sharing) and unzip them under the `Visually-Imabalanced-Stereo` folder.  The models serve as pre-trained models for dynamic filters and pre-trained models to compute the preceptual loss.

5. Prepare the stereo matching network following instruction at [CRL](https://github.com/Artifineuro/crl).

#### Training and Testing
1. To start training and testing, run `run.sh` . We pre-select the horizontal filter size to be 201, which is based on actual image width and maximum disparity in the dataset.
2. We have a toy dataset under the repo as to show the example layout. Alternatively, you could use KITTI stereo 2015 or your own dataset to train the view restoration network.
3. After getting the restored right image, load the pre-trained checkpoints of the stereo matching network CRL, and compute the disparity predictions.

