import cv2
import math
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image, ImageOps
from transform import Relabel, ToLabel
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import ColorJitter, RandomCrop

# ========== ERFNET DATA AUGMENTATION ==========
class ErfNetTransform(object):
    """
    Data augmentation for training ERFNet model, including resize on both input and 
    target image, horizontal flip and translations.
    """
    def __init__(self, enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height

    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Horizontal flip
            if (random.random() < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random translation 0-2 pixels (fill rest with padding)
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
        target = Relabel(255, 19)(target)

        return input, target

# ========== BISENET DATA AUGMENTATION (MEMORY-SAFE) ==========   
class BiSeNetTransform(object):
    """
    Data augmentation for training BiSeNet model, including random resized crop, 
    horizontal flip and color jitter. Inspired by the augmentation strategy used in: 
    https://github.com/CoinCheung/BiSeNet/blob/master/lib/data/transform_cv2.py.

    NOTE: Augmentations have been rewritten using PIL and torchvision to be memory-safe 
    (no OpenCV or NumPy as in the official version), reducing the risk of CUDA OOM errors.
    """
    def __init__(self, augment=True, height=512, scales=(0.75, 1.25)):  # 2.0
        self.augment = augment
        self.height = height
        self.scales = scales
        self.color_jitter = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)

    def __call__(self, input, target):
        # Resize input and target proportionally to a scaled height
        scale = random.uniform(*self.scales)
        scaled_height = int(self.height * scale)
        input = Resize(scaled_height, Image.BILINEAR)(input)
        target = Resize(scaled_height, Image.NEAREST)(target)

        if self.augment:
            # Pad if needed for following crop - scale < 1.0
            pad_h = max(0, self.height - input.height)
            pad_w = max(0, self.height - input.width)

            if pad_h > 0 or pad_w > 0:
                input = TF.pad(input, padding=(0, 0, pad_w, pad_h), fill=0)
                target = TF.pad(target, padding=(0, 0, pad_w, pad_h), fill=255)

            # Random crop synchronized for input and target
            i, j, h, w = RandomCrop.get_params(input, output_size=(self.height, self.height))
            input = TF.crop(input, i, j, h, w)
            target = TF.crop(target, i, j, h, w)

            # Horizontal flip
            if random.random() < 0.5:
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            # Color jitter (only on input)
            input = self.color_jitter(input)

        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target
    
# ========== ENET DATA AUGMENTATION ==========
class ENetTransform(object):
    """
    Data augmentation for training ENet model, including resize on both input and
    target image, horizontal flip, translations, brightness adjustment and random noise.
    """
    def __init__(self, augment=True, height=512):
        self.augment = augment
        self.height = height

    def __call__(self, input, target):
        input = Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            # Random translation   
            transX = random.randint(-1, 1)
            transY = random.randint(-1, 1)
            
            input = ImageOps.expand(input, border=(transX, transY, 0, 0), fill=0)
            target = ImageOps.expand(target, border=(transX, transY, 0, 0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))

            # Random brightness adjustment
            brightness_factor = random.uniform(0.6, 1.2)  # brightness between 60% and 120%
            input = TF.adjust_brightness(input, brightness_factor)

            # Random noise
            noise = np.random.normal(0, 2, np.array(input).shape).astype(np.uint8)  # gaussian noise
            input = np.clip(np.array(input) + noise, 0, 255).astype(np.uint8)
            input = Image.fromarray(input)
            
        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target