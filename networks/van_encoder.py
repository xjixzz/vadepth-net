from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from .van import *
from timm.models import create_model
from .van import OverlapPatchEmbed


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def van_multiimage_input(size_encoder, pretrained=False, num_input_images=1):
    """Constructs a VAN model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['patch_embed1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model

    
class VANEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, size_encoder, pretrained, num_input_images=1):
        super(VANEncoder, self).__init__()
        
        van_paths = {"tiny": "./pretrained/van_tiny_754.pth.tar",
        #van_paths = {"tiny": "./pretrained/van_tiny_seg.pth",
                #"small": "./pretrained/van_small_seg.pth",
                "small": "./pretrained/van_small_811.pth.tar",
                "base": "./pretrained/van_base_828.pth.tar"
                }
        if size_encoder=='tiny':
            self.num_ch_enc = np.array([32, 64, 160, 256])
        else:
            self.num_ch_enc = np.array([64, 128, 320, 512])
            
        if size_encoder not in van_paths:
            raise ValueError("{} is not a valid size of vnn".format(size_encoder))

        if num_input_images > 1:
            #self.encoder = van_multiimage_input(num_layers, pretrained, num_input_images)
            self.encoder = create_model("van_{}".format(size_encoder), pretrained=False, num_classes=None, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None)

            self.encoder.patch_embed1 = OverlapPatchEmbed(in_chans=3*num_input_images, embed_dim = self.num_ch_enc[0])
            if pretrained:
                pretrined_state_dict = torch.load(van_paths[size_encoder])['state_dict']
                pretrined_state_dict['patch_embed1.proj.weight'] = torch.cat(
                    [pretrined_state_dict['patch_embed1.proj.weight']] * num_input_images, 1) / num_input_images
                self.encoder.load_state_dict(pretrined_state_dict)
        else:
            self.encoder = create_model("van_{}".format(size_encoder), pretrained=False, num_classes=None, drop_rate=0.0, drop_path_rate=0.1, drop_block_rate=None)
            
            if pretrained:
                encoder_dict = self.encoder.state_dict()
                pretrined_state_dict = torch.load(van_paths[size_encoder])['state_dict']
                load_state_dict = {k:v for k,v in pretrined_state_dict.items() if k in encoder_dict.keys()}
                encoder_dict.update(load_state_dict)
                self.encoder.load_state_dict(encoder_dict)
                


    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        
        B = x.shape[0]

        for i in range(4):
            patch_embed = getattr(self.encoder, f"patch_embed{i + 1}")
            block = getattr(self.encoder, f"block{i + 1}")
            norm = getattr(self.encoder, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                #x = blk(x, H, W)
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            #if i != 3:
                #x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                
            self.features.append(x)
            
            #print(x.shape)
            

        return self.features
