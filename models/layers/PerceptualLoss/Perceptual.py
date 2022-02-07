from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch

from . import vgg

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


class vgg19_wrapper(nn.Module):
    def __init__(self, pretrained=False):
        super(vgg19_wrapper, self).__init__()
        vgg19 = vgg.network(vgg.make_layers(vgg.cfg['vgg19']), fix_weights=True)
        self.layers = vgg19.features
        if pretrained:
            vgg19.load_state_dict(torch.load('./weights/vgg19.pth'))
            print ("===> VGG19 Wrapper: Using Pretrained Weights.")
            self.layers = vgg19.features
        self.norm_stats = {'mean': (0.485, 0.456, 0.406),
                           'std' : (0.229, 0.224, 0.225)}

    def forward(self, x, layer_ids):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.layers._modules.items():
            x = layer(x)
            if name in layer_ids:
                features.append(x)
        return features

class loss(torch.nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.moduleNormalize = std_norm()

    def forward(self, img_pred_tensor, img_gt_tensor, wrapper, layer_ids):

        pred = self.moduleNormalize(img_pred_tensor, wrapper.norm_stats['mean'], wrapper.norm_stats['std'])
        gt = self.moduleNormalize(img_gt_tensor, wrapper.norm_stats['mean'], wrapper.norm_stats['std'])

        pred_features = wrapper(pred, layer_ids)
        gt_features = wrapper(gt, layer_ids)

        loss_val = 0
        for fp, fg in zip(pred_features, gt_features):
            loss_val += torch.mean((fp - fg) ** 2)

        loss_val /= len(pred_features)
        return loss_val
