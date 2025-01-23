import cv2
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image, ImageOps
from transform import Relabel, ToLabel
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Pad, RandomCrop


# Augmentations
class ErfNetTransform(object):
    """
    Different functions implemented to perform random augments on both image 
    and target for ErfNet model (e.g. resize, horizontal flip, traslations, ...)
    """
    def __init__(self, enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass

    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        target = ToLabel()(target)


class BiSeNetTransform(object):
    """
    Different functions implemented to perform random augments on both image 
    and target for BiSeNet model (e.g. mean subtraction, horizontal flip, scale, 
    crop into fix size). As suggested in the BiSeNet official paper.
    """
    def __init__(self, crop_size = (512, 512)):
        self.crop_size = crop_size
        self.scales = [0.75, 1.0, 1.5, 1.75, 2.0]
        self.mean = np.array([0.287, 0.325, 0.284])

    def __call__(self, input, target):
        input = Resize(self.crop_size[0], Image.BILINEAR, antialias=True)(input)
        target = Resize(self.crop_size[0], Image.NEAREST, antialias=True)(target)

        # Mean substraction
        input = np.asarray(input).astype(np.float32) / 255.0
        input = input - self.mean

        input = torch.from_numpy(input.transpose(2, 0, 1)).float()

        # Random horizontal flip
        if random.random() > 0.5:
            input = TF.hflip(input)
            target = TF.hflip(target)

        # Apply scale
        scale = random.choice(self.scales)
        size = int(scale * self.crop_size[0])
        input = Resize(size, Image.BILINEAR, antialias=True)(input)
        target = Resize(size, Image.NEAREST, antialias=True)(target)
        if scale == 0.75:
            padding = 128, 128
            input = Pad(padding, fill=0, padding_mode='constant')(input)
            target = Pad(padding, fill=255, padding_mode='constant')(target)
        
        # Crop
        i, j, h, w = RandomCrop.get_params(input, output_size=(self.crop_size[0], self.crop_size[1]))
        input = TF.crop(input, i, j, h, w)
        target = TF.crop(target, i, j, h, w)

        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target
    

class ENetTransform(object):
    """
    Different functions implemented to perform random augments on both image 
    and target for ENet model (e.g. resize, horizontal flip, rotations, ...)
    """
    def __init__(self, augment=True, height=512):
        self.augment = augment
        self.height = height

    def __call__(self, input, target):
        input = Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if self.augment:
            # Random hflip
            hflip = random.random()
            if hflip < 0.5:
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            # Random rotation between -10 and 10 degrees
            angle = random.uniform(-10, 10)
            input = input.rotate(angle, resample=Image.BILINEAR)
            target = target.rotate(angle, resample=Image.NEAREST)

        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)
        return input, target