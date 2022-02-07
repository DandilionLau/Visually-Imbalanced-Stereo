#!/usr/bin/env python2.7
import os
import argparse
import sys
import getopt
import math
import time
import numpy
import PIL
import torch
import torchvision
import torch.utils.serialization
from torch.utils.data import DataLoader

from models import *
from utils import *

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

##########################################################
parser = argparse.ArgumentParser(description='Kernel Prediction Network')
parser.add_argument('--dataset', default = 'KITTI', required=False, help='Default KITTI')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--filter_size_horizontal', type=int, default=301, help='horizontal size of 1D filter')
parser.add_argument('--filter_size_vertical', type=int, default=0, help='vertical size of 1D filter')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels, default=3')
parser.add_argument('--image_height', type=int, default=375, help='height of input image, default=375 for KITTI')
parser.add_argument('--image_width', type=int, default=1242, help='width of input image, default=1242 for KITTI')
parser.add_argument('--output_height', type=int, default=375, help='height of output image, default=375 for KITTI')
parser.add_argument('--output_width', type=int, default=1242, help='width of output image, default=1242 for KITTI')
parser.add_argument('--scale_factor', type=int, default=10, help='scale factor for downgrading the right view, default=5')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size, default=1')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--save_dir', default = './result', required=False, help='default as test')
parser.add_argument('--checkpoint_dir', default = './checkpoint', required=False, help='default as checkpoint')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--gpu_num', type=int, default=1, help='number of gpu you want to use')
parser.add_argument('--loading_weights', type=int, default=1, help='whether loading from existing weights')
parser.add_argument('--weight_source', type=str, default='std', help='ours or std, defualt is std')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam, AdaMax or SGD, default is Adam')
parser.add_argument('--loss', type=str, default='Smooth-L1', help='L1 or L2, default is L2')
parser.add_argument('--only_test', type=int, default=0, help='1 for only testing with existing model')
opt = parser.parse_args()

if(opt.gpu_num == 1):
    torch.cuda.device(0)
else:
    torch.cuda.device(range(opt.gpu_num+1))

print('Where Am I')

perceptual_layers = ['0', '5', '10', '19', '28']
norm_stats = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
VGG19_Net = Perceptual.vgg19_wrapper(pretrained=True).cuda()

for child in VGG19_Net.children():
    for name, param in VGG19_Net.named_parameters():
            param.requires_grad = False
            print('===> Layer Fixed: ' + name)

moduleUnnormalize = std_norm(inverse=True).cuda()
PerceptualLoss = Perceptual.loss().cuda()

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.9
        ("===> Update learning rate:" + str(param_group['lr']))

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        print('===> Initialize Layer ' + str(m) + ' by Normal Distribution')

    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        print('===> Initialize Layer ' + m.name + ' with Zero')

print('===> Loading Datasets')
root_path = "./dataset/"
train_set = get_training_set(root_path + opt.dataset, opt.image_height, opt.image_width, scale_factor=opt.scale_factor, input_nc=opt.input_nc)
test_set = get_test_set(root_path + opt.dataset, opt.image_height, opt.image_width, scale_factor=opt.scale_factor, input_nc=opt.input_nc)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building KPN Model')

if (opt.loss == 'L1'):
    critirion = torch.nn.L1Loss(size_average=True,reduce=True).cuda()
elif (opt.loss == 'L2'):
    critirion = torch.nn.MSELoss(size_average=True,reduce=True).cuda()
elif(opt.loss == 'Smooth-L1'):
    critirion = torch.nn.SmoothL1Loss(size_average=True,reduce=True).cuda()

msssim_loss = MSSSIM().cuda()

print('===> Initialize Matched Weights')

if(opt.loading_weights == 1):
    if(opt.weight_source == 'std'):
        moduleNetwork = KPN_VIS(opt.filter_size_vertical,opt.filter_size_horizontal)
        stdNetwork = KPN_std(opt.filter_size_vertical,opt.filter_size_horizontal)

        if(opt.gpu_num > 1):
            moduleNetwork = torch.nn.DataParallel(moduleNetwork,device_ids=range(opt.gpu_num))

        moduleNetwork = moduleNetwork.cuda()
        stdNetwork = stdNetwork.cuda()

        stdNetwork.load_state_dict(torch.load('./weights/network-lf.pytorch'))

        ours_dict = moduleNetwork.state_dict()
        std_dict = stdNetwork.state_dict()

        if(opt.gpu_num > 1):
            from collections import OrderedDict
            modified_state_dict = OrderedDict()

            for k, v in std_dict.items():
                namekey = 'module.' + k # add `module.`
                modified_state_dict[namekey] = v

            std_dict = modified_state_dict

        # 1. filter out unnecessary keys
        std_dict = {k: v for k, v in ours_dict.items() if k in std_dict}
        # 2. overwrite entries in the existing state dict
        ours_dict.update(std_dict)

        for layer in ours_dict.keys():
            if(layer in std_dict.keys()):
                print('===> Loading Weight for Layer ' + layer)

        not_initialized_layers = list(set(moduleNetwork.state_dict().keys()).difference(set(std_dict.keys())))

        if (not_initialized_layers != []):
            for layer in not_initialized_layers:
                print('===> Layer Not Initailized: ' + layer)

        # 3. load the new state dict
        moduleNetwork.load_state_dict(ours_dict)

    elif(opt.weight_source == 'ours'):
        moduleNetwork = KPN_VIS(opt.filter_size_vertical,opt.filter_size_horizontal)
        old_state_dict = torch.load('KPN.pth')
        moduleNetwork.load_state_dict(old_state_dict, strict=False)
        moduleNetwork.cuda()
        print("===> Loading Weight from Previous Training")

elif(opt.loading_weights == 0):
    moduleNetwork = KPN_VIS(opt.filter_size_vertical,opt.filter_size_horizontal)
    moduleNetwork.apply(weights_init)
    moduleNetwork = moduleNetwork.cuda()

if(opt.optimizer == 'AdaMax'):
    optimizer = torch.optim.Adamax(moduleNetwork.parameters(), lr=opt.lr, betas=(0.9, 0.999))
elif(opt.optimizer == 'SGD'):
    optimizer = torch.optim.SGD(moduleNetwork.parameters(), lr=opt.lr, momentum=0.9)
elif(opt.optimizer == 'Adam'):
    optimizer = torch.optim.Adam(moduleNetwork.parameters(), lr=opt.lr, betas=(0.5, 0.999))

if(opt.gpu_num > 1):
    optimizer = torch.nn.DataParallel(optimizer, device_ids=range(opt.gpu_num+1))

intPaddingLeft = int(math.floor(opt.filter_size_horizontal / 2.0))
intPaddingRight = int(math.floor(opt.filter_size_horizontal / 2.0))
intPaddingTop = int(math.floor(opt.filter_size_vertical / 2.0))
intPaddingBottom = int(math.floor(opt.filter_size_vertical / 2.0))
modulePaddingInput = torch.nn.Sequential()
modulePaddingOutput = torch.nn.Sequential()

if True:
    intPaddingWidth = intPaddingLeft + opt.image_width + intPaddingRight
    intPaddingHeight = intPaddingTop + opt.image_height + intPaddingBottom

    if intPaddingWidth != ((intPaddingWidth >> 7) << 7):
        intPaddingWidth = (((intPaddingWidth >> 7) + 1) << 7) # more than necessary
    # end

    if intPaddingHeight != ((intPaddingHeight >> 7) << 7):
        intPaddingHeight = (((intPaddingHeight >> 7) + 1) << 7) # more than necessary
    # end

    intPaddingWidth = intPaddingWidth - (intPaddingLeft + opt.image_width + intPaddingRight)
    intPaddingHeight = intPaddingHeight - (intPaddingTop + opt.image_height + intPaddingBottom)

    modulePaddingInput = modulePaddingInput.cuda()
    modulePaddingOutput = modulePaddingOutput.cuda()


def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):
        left_view_cpu, right_view_cpu, impaired_view_cpu, filename = batch[0], batch[1], batch[2], batch[3]

        modulePaddingInput = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight + intPaddingWidth, intPaddingTop, intPaddingBottom + intPaddingHeight])
        modulePaddingOutput = torch.nn.ReplicationPad2d([0 - intPaddingLeft, 0 - intPaddingRight - intPaddingWidth, 0 - intPaddingTop, 0 - intPaddingBottom - intPaddingHeight])

        modulePaddingInput = modulePaddingInput.cuda()
        modulePaddingOutput = modulePaddingOutput.cuda()

        variablePaddingFirst = modulePaddingInput(torch.autograd.Variable(left_view_cpu, requires_grad=True)).cuda()
        variablePaddingSecond = modulePaddingInput(torch.autograd.Variable(impaired_view_cpu, requires_grad=True)).cuda()

        variablePaddingFirst = variablePaddingFirst.cuda()
        variablePaddingSecond = variablePaddingSecond.cuda()

        optimizer.zero_grad()

        variablePaddingOutput = moduleNetwork(variablePaddingFirst, variablePaddingSecond)
        variablePaddingOutput = modulePaddingOutput(variablePaddingOutput)

        right_view_gt = torch.autograd.Variable(right_view_cpu, requires_grad=False).cuda()
        right_view_pred = variablePaddingOutput.cuda()

        loss_texture = critirion(right_view_pred,right_view_gt)
        loss_structure = 1-msssim_loss(right_view_pred,right_view_gt)

        pred_tensor = moduleUnnormalize(right_view_pred, norm_stats['mean'], norm_stats['std'])
        gt_tensor = moduleUnnormalize(right_view_cpu.cuda(), norm_stats['mean'], norm_stats['std'])
        loss_perceptual = PerceptualLoss(pred_tensor, gt_tensor, VGG19_Net, perceptual_layers)

        if(torch.isnan(loss_structure)):
            loss = 1 * (0.5 * loss_texture + 0.5 * loss_perceptual)
        else:
            loss = 1 * (0.42 * loss_texture + 0.08 * loss_structure + 0.5 * loss_perceptual)

        loss.backward()

        if(opt.gpu_num == 1):
            optimizer.step()
        elif(opt.gpu_num > 1):
            optimizer.module.step()

        #print("===> Training: Epoch[{}]({}/{}): \n Loss: Pixel:[{:.4f}] SSIM:[{:.4f}] Percetual:[{:.4f}]".format(
        #      epoch, iteration, len(training_data_loader), loss_texture.item(), 1-loss_structure.item(), loss_perceptual.item()))

def test(epoch):
    tic = time.clock()
    test_img_count = 1

    avg_psnr = 0
    avg_msssim = 0

    with torch.no_grad():
        for batch in testing_data_loader:
            left_view_cpu, right_view_cpu, impaired_view_cpu, filename = batch[0], batch[1], batch[2], batch[3]
            

            modulePaddingInput = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight + intPaddingWidth, intPaddingTop, intPaddingBottom + intPaddingHeight])
            modulePaddingOutput = torch.nn.ReplicationPad2d([0 - intPaddingLeft, 0 - intPaddingRight - intPaddingWidth, 0 - intPaddingTop, 0 - intPaddingBottom - intPaddingHeight])


            modulePaddingInput = modulePaddingInput.cpu()
            modulePaddingOutput = modulePaddingOutput.cpu()

            variablePaddingFirst = modulePaddingInput(torch.autograd.Variable(left_view_cpu, requires_grad=True)).cuda()
            variablePaddingSecond = modulePaddingInput(torch.autograd.Variable(impaired_view_cpu, requires_grad=True)).cuda()

            variablePaddingFirst = variablePaddingFirst.cuda()
            variablePaddingSecond = variablePaddingSecond.cuda()
            
            print(variablePaddingFirst.size(), variablePaddingSecond.size())


            # ModuleNetwork
            variablePaddingOutput = moduleNetwork(variablePaddingFirst, variablePaddingSecond)


            print('SeparableConvolution output: ', variablePaddingOutput.size())
            
            #variablePaddingOutput = modulePaddingOutput(variablePaddingOutput)
            #print('SeparableConvolution pad output: ', variablePaddingOutput.size())


            right_view_gt = torch.autograd.Variable(right_view_cpu).cpu()


            right_view_pred = variablePaddingOutput.cpu()

            print(right_view_gt.size())

            # Here 
            avg_psnr += 10 * math.log10(1 / critirion(right_view_pred, right_view_gt).item())
            avg_msssim += msssim_loss(right_view_pred,right_view_gt).item()

            if not os.path.exists(os.path.join("result", opt.dataset)):
                os.makedirs(os.path.join("result", opt.dataset))

            for i in range(0,opt.testBatchSize):
                save_img(torch.autograd.Variable(left_view_cpu).data[i], "result/{}/{}".format(opt.dataset, 'epoch_'+str(epoch)+'_'+str(test_img_count)+'_left_good'+'.jpg'),opt.output_height,opt.output_width)
                save_img(torch.autograd.Variable(right_view_cpu).data[i], "result/{}/{}".format(opt.dataset, 'epoch_'+str(epoch)+'_'+str(test_img_count)+'_right_good'+'.jpg'),opt.output_height,opt.output_width)
                save_img(torch.autograd.Variable(impaired_view_cpu).data[i], "result/{}/{}".format(opt.dataset, 'epoch_'+str(epoch)+'_'+str(test_img_count)+'_right_bad'+'.jpg'),opt.output_height,opt.output_width)
                save_img(right_view_pred.data[i], "result/{}/{}".format(opt.dataset, 'epoch_'+str(epoch)+'_'+str(test_img_count)+'_right_syn'+'.jpg'), opt.output_height,opt.output_width)
                #save_img(right_view_ref.data[i],  "result/{}/{}".format(opt.dataset, 'epoch_'+str(epoch)+'_'+str(test_img_count)+'_right_ref'+'.jpg'), opt.output_height,opt.output_width)
                test_img_count += 1

            # Release memory
            del right_view_pred, right_view_gt

            if(test_img_count > 11):
                if(opt.only_test != 1):
                    break

        '''
        if(test_img_count >= len(testing_data_loader)*opt.testBatchSize):
            print("----------------------------Testing--------------------------------")
            print("===> Testing: Epoch[{}]: loss_texture: {:.4f} ".format(
            epoch, test_texture_loss*opt.image_height*opt.image_width*opt.input_nc*opt.testBatchSize/test_img_count))
            print("===> Testing: Epoch[{}]: loss_structure: {:.4f} ".format(
                epoch, test_structure_loss*opt.image_height*opt.image_width*opt.input_nc*opt.testBatchSize/opt.scale_factor/opt.scale_factor/test_img_count))
            print("----------------------------Testing--------------------------------")

            test_texture_loss = 0
        '''
        toc = time.clock()
        print('===> Avg. Time: '+ str((toc-tic)/(test_img_count-1))+' sec')
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / (test_img_count-1)))
        print("===> Avg. MS-SSIM: {:.4f} ".format(avg_msssim / (test_img_count-1)))


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))

    KPN_out_path = "checkpoint/{}/KPN_epoch_{}.pth".format(opt.dataset, epoch)
    torch.save(moduleNetwork.state_dict(),KPN_out_path)
    print("Checkpoint saved to {}".format("checkpoint/" + opt.dataset))



for epoch in range(1, opt.nEpochs + 1):

    if(opt.only_test == 1):
        test(epoch)
        exit()

    test(epoch)
    train(epoch)

    if epoch % 10 == 0:
        checkpoint(epoch)
    if epoch % 3 == 0:
        adjust_learning_rate(optimizer,epoch)
