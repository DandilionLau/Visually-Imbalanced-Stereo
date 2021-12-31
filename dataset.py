import numpy
from os import listdir
from os.path import join
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

imagenet_stats = {'mean': [0.5, 0.5, 0.5],
           'std': [0.5, 0.5, 0.5]}

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, input_nc, is_disp = False, image_height=None, image_width=None):
    if(is_disp == False):
        img = Image.open(filepath).convert('RGB')
    else:
        img = Image.open(filepath)
        img = img.resize((image_width,image_height))
        img = numpy.ascontiguousarray(img,dtype=numpy.float32)/256*(1242/image_width)
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, height, width, scale, input_nc, if_test=0, normalize=imagenet_stats):
        super(DatasetFromFolder, self).__init__()
        self.left_view = join(image_dir, "image_2")
        self.right_view = join(image_dir, "image_3")
        #self.distort_view = join(image_dir, "image_3_gen")
        self.input_nc = input_nc
        self.if_test = if_test
        self.height = height
        self.width = width

        try:
            FileNotFoundError
        except NameError:
            FileNotFoundError = IOError

        self.image_filenames = [x for x in listdir(self.left_view) if is_image_file(x)]

        transform_list = [transforms.Resize(size=(height,width)),transforms.ToTensor(),transforms.Normalize(**normalize)]
        transform_list_scale = [transforms.Resize(size=(int(height/scale),int(width/scale))),transforms.Resize(size=(height,width)),transforms.ToTensor(),transforms.Normalize(**normalize)]

        self.transform = transforms.Compose(transform_list)
        self.transform_scale = transforms.Compose(transform_list_scale)

    def __getitem__(self, index):
        # Load Image
        file_name = self.image_filenames[index]
        left = load_img(join(self.left_view, self.image_filenames[index]), self.input_nc)
        right = load_img(join(self.right_view, self.image_filenames[index]), self.input_nc)
        #distort =  load_img(join(self.distort_view, self.image_filenames[index]), self.input_nc)

        left_img = self.transform(left)
        right_img = self.transform(right)
        imparied_img = self.transform_scale(right)

        return left_img, right_img, imparied_img, file_name

    def __len__(self):
        return len(self.image_filenames)
