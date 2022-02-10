import math
import torch
import PIL
import operator

from .layers import *
from guided_filter_pytorch.guided_filter import GuidedFilter

class std_norm(nn.Module):
    def __init__(self, inverse=False):
        super(std_norm, self).__init__()
        self.inverse = inverse

    def forward(self, x, mean, std):
        # x: [N, C, H, W]
        out = []
        for i in range(len(mean)):
            if not self.inverse:
                normalized = (x[:, i, :, :] - mean[i]) / std[i]
            else:
                normalized = x[:, i, :, :] * std[i] + mean[i]
            normalized = torch.unsqueeze(normalized, 1)
            out.append(normalized)
        return torch.cat(out, dim=1)



class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x



class KPN_VIS(torch.nn.Module):
    def __init__(self, filter_size_vertical, filter_size_horizontal):
        self.filter_size_vertical = filter_size_vertical
        self.filter_size_horizontal = filter_size_horizontal
        super(KPN_VIS, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Subnet_Horizontal():
            if(filter_size_horizontal == 0):
                return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    #torch.nn.UpsamplingBilinear2d(scale_factor=2),
                    torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
                )
            elif(filter_size_horizontal > 0):
                return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=filter_size_horizontal, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    #torch.nn.UpsamplingBilinear2d(scale_factor=2),
                    torch.nn.Conv2d(in_channels=filter_size_horizontal, out_channels=filter_size_horizontal, kernel_size=3, stride=1, padding=1)
                )

        def Subnet_Vertical():
            if(filter_size_vertical == 0):
                return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    #torch.nn.UpsamplingBilinear2d(scale_factor=2),
                    torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
                )
            elif(filter_size_vertical > 0):
                return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=filter_size_vertical, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    #torch.nn.UpsamplingBilinear2d(scale_factor=2),
                    torch.nn.Conv2d(in_channels=filter_size_vertical, out_channels=filter_size_vertical, kernel_size=3, stride=1, padding=1)
                )
        # end

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = torch.nn.Sequential(
            #torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = torch.nn.Sequential(
            #torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = torch.nn.Sequential(
            #torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = torch.nn.Sequential(
            #torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVertical_new1 = Subnet_Vertical()
        self.moduleHorizontal_new1 = Subnet_Horizontal()
        self.guided_filter = GuidedFilter(5,1e-2)

        self.modulePad = torch.nn.ReplicationPad2d([ int(math.floor(filter_size_horizontal / 2.0)), int(math.floor(filter_size_horizontal / 2.0)), int(math.floor(filter_size_vertical / 2.0)), int(math.floor(filter_size_vertical / 2.0)) ])


        self.additional_dot_kernal = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 5), stride=1, padding=1)
        self.additional_ada_pool = nn.AdaptiveMaxPool2d((375,1242))
        
        # Separable Conv
    
    # end

    #def forward(self, variableInput1, variableInput2):
    def forward(self, variableInput1, variableInput2):
        def rgb2gray(rgb_tensor):
            gray_tensor = (0.2989 * rgb_tensor[:,0,:,:] + 0.5870 * rgb_tensor[:,1,:,:] + 0.1140 * rgb_tensor[:,1,:,:]).unsqueeze(1)
            return gray_tensor
        #variableJoin = variableInput1

        variableJoin = torch.cat([variableInput1,variableInput2], 1)
        #print(variableJoin.size)

        variableConv1 = self.moduleConv1(variableJoin)
        variablePool1 = self.modulePool1(variableConv1)

        variableConv2 = self.moduleConv2(variablePool1)
        variablePool2 = self.modulePool2(variableConv2)

        variableConv3 = self.moduleConv3(variablePool2)
        variablePool3 = self.modulePool3(variableConv3)

        variableConv4 = self.moduleConv4(variablePool3)
        variablePool4 = self.modulePool4(variableConv4)

        variableConv5 = self.moduleConv5(variablePool4)
        variablePool5 = self.modulePool5(variableConv5)

        variableDeconv5 = self.moduleDeconv5(variablePool5)
        variableUpsample5 = self.moduleUpsample5(variableDeconv5)

        variableCombine = variableUpsample5 + variableConv5

        variableDeconv4 = self.moduleDeconv4(variableCombine)
        variableUpsample4 = self.moduleUpsample4(variableDeconv4)

        variableCombine = variableUpsample4 + variableConv4

        variableDeconv3 = self.moduleDeconv3(variableCombine)
        variableUpsample3 = self.moduleUpsample3(variableDeconv3)

        variableCombine = variableUpsample3 + variableConv3

        variableDeconv2 = self.moduleDeconv2(variableCombine)
        variableUpsample2 = self.moduleUpsample2(variableDeconv2)

        variableCombine = variableUpsample2 + variableConv2


        horizontal_kernels = self.moduleHorizontal_new1(variableCombine)

        n,c,h,w = horizontal_kernels.size()
        max_mean = 0
        max_c = 0
        for i in range(0, c):
            mean = horizontal_kernels[:,i,:,:].mean().abs()
            if mean > max_mean:
                max_mean = mean
                max_c = i


        horizontal_kernels[:,max_c,:,:] = self.guided_filter(rgb2gray(variableInput1), horizontal_kernels[:,max_c,:,:].clone().unsqueeze(1))

        # Original Code
        variableDot2 = SeparableConvolution(self.filter_size_vertical,  self.filter_size_horizontal)(self.modulePad(variableInput1), self.moduleVertical_new1(variableCombine), horizontal_kernels)
        variableDot2 = self.moduleVertical_new1(variableCombine)
        
        
        # If 'SeparableConv' doesnt work, this is a temporal new replacement
        #variableDot2 = self.moduleVertical_new1(variableCombine)
        #variableDot2 = self.additional_dot_kernal(variableDot2)
        #variableDot2 = self.additional_ada_pool(variableDot2)

        return  variableDot2
    # end
# end



class KPN_std(torch.nn.Module):
    def __init__(self, filter_size_vertical, filter_size_horizontal):
        self.filter_size_vertical = filter_size_vertical
        self.filter_size_horizontal = filter_size_horizontal
        super(KPN_std, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )
        # end

        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                #torch.nn.UpsamplingBilinear2d(scale_factor=2),
                torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
            )
        # end

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVertical1 = Subnet()
        self.moduleVertical2 = Subnet()
        self.moduleHorizontal1 = Subnet()
        self.moduleHorizontal2 = Subnet()

        self.modulePad = torch.nn.ReplicationPad2d([ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ])

        #self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
    # end

    def forward(self, variableInput1, variableInput2):
        variableJoin = torch.cat([variableInput1, variableInput2], 1)

        variableConv1 = self.moduleConv1(variableJoin)
        variablePool1 = self.modulePool1(variableConv1)

        variableConv2 = self.moduleConv2(variablePool1)
        variablePool2 = self.modulePool2(variableConv2)

        variableConv3 = self.moduleConv3(variablePool2)
        variablePool3 = self.modulePool3(variableConv3)

        variableConv4 = self.moduleConv4(variablePool3)
        variablePool4 = self.modulePool4(variableConv4)

        variableConv5 = self.moduleConv5(variablePool4)
        variablePool5 = self.modulePool5(variableConv5)

        variableDeconv5 = self.moduleDeconv5(variablePool5)
        variableUpsample5 = self.moduleUpsample5(variableDeconv5)

        variableCombine = variableUpsample5 + variableConv5

        variableDeconv4 = self.moduleDeconv4(variableCombine)
        variableUpsample4 = self.moduleUpsample4(variableDeconv4)

        variableCombine = variableUpsample4 + variableConv4

        variableDeconv3 = self.moduleDeconv3(variableCombine)
        variableUpsample3 = self.moduleUpsample3(variableDeconv3)

        variableCombine = variableUpsample3 + variableConv3

        variableDeconv2 = self.moduleDeconv2(variableCombine)
        variableUpsample2 = self.moduleUpsample2(variableDeconv2)

        variableCombine = variableUpsample2 + variableConv2

        variableDot1 = SeparableConvolution(self.filter_size_vertical,self.filter_size_horizontal)(self.modulePad(variableInput1), self.moduleVertical1(variableCombine), self.moduleHorizontal1(variableCombine))
        variableDot2 = SeparableConvolution(self.filter_size_vertical,self.filter_size_horizontal)(self.modulePad(variableInput2), self.moduleVertical2(variableCombine), self.moduleHorizontal2(variableCombine))

        return variableDot1 + variableDot2
