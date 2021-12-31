import math
import torch
import skimage
import skimage.io
import numpy as np
import PIL.Image as Image
from os.path import join
from dataset import DatasetFromFolder
from torchvision import transforms

from skimage.transform import resize

def get_training_set(root_dir,image_height,image_width,input_nc, scale_factor):
    train_dir = join(root_dir, "train")
    return DatasetFromFolder(train_dir,image_height,image_width, scale_factor, input_nc)

def get_test_set(root_dir,image_height,image_width, input_nc, scale_factor):
    test_dir = join(root_dir, "test")
    return DatasetFromFolder(test_dir,image_height,image_width,scale_factor, input_nc, if_test=1)

def save_img(image_tensor, filename, output_height, output_width, is_disp=0):
    if(is_disp == 0):
        image_numpy = image_tensor.numpy()
        if(image_numpy.max()>1) or (image_tensor.min()<-1):
            image_numpy = np.maximum(image_numpy,np.full_like(image_numpy,-1))
            image_numpy = np.minimum(image_numpy,np.full_like(image_numpy,1))

        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2 * 255.0
        image_numpy = image_numpy.astype(np.uint8)
        image_pil = Image.fromarray(image_numpy)
        image_pil = image_pil.resize((output_width,output_height))
        image_pil.save(filename)
    else:
        if(image_tensor.max() > 65535):
            image_tensor = np.minimum(image_tensor,np.full_like(image_tensor,65535))
        if(image_tensor.min() < 0):
            image_tensor = np.maximum(image_tensor,np.full_like(image_tensor,0))
        image_tensor = resize(image_tensor.astype('uint16'),(output_height,output_width), mode='constant')
        image_numpy = (image_tensor*65535).astype('uint16')
        if(image_tensor.max() > 65535):
            image_tensor = np.minimum(image_tensor,np.full_like(image_tensor,65535))
        if(image_tensor.min() < 0):
            image_tensor = np.maximum(image_tensor,np.full_like(image_tensor,0))
        skimage.io.imsave(filename,image_numpy, plugin='freeimage')

    #print("Image saved as {}".format(filename))

def Resize2D(img,size):
    return (torch.nn.functional.upsample(img,size, align_corners=True, mode='bilinear')).data
