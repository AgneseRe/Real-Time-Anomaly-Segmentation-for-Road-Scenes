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
from torchvision.transforms import Pad, RandomCrop

# ========== ERFNET DATA AUGMENTATION ==========
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
        target = Relabel(255, 19)(target)

        return input, target

# ========== BISENET DATA AUGMENTATION ==========   
# Source: https://github.com/CoinCheung/BiSeNet/blob/master/lib/data/transform_cv2.py
class BiSeNetTransformTrain(object):

    def __init__(self, scales=(0.75, 2.0), cropsize=(1024, 1024)):
        self.trans_func = Compose([
            RandomResizedCrop(scales, cropsize),
            RandomHorizontalFlip(),
            ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        ])

    def __call__(self, input, target):
        input, target = self.trans_func(input, target)
        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)
        return input, target
    
class BiSeNetTransformVal(object):

    def __call__(self, input, target):
        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)
        return input, target

class RandomResizedCrop(object):
    '''
    size should be a tuple of (H, W)
    '''
    def __init__(self, scales=(0.5, 1.), size=(384, 384)):
        self.scales = scales
        self.size = size

    def __call__(self, im, lb):
        if self.size is None:
            return im, lb

        if not isinstance(im, np.ndarray):
            im = np.array(im)
        if not isinstance(lb, np.ndarray):
            lb = np.array(lb)
        assert im.shape[:2] == lb.shape[:2]

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))
        im_h, im_w = [math.ceil(el * scale) for el in im.shape[:2]]
        im = cv2.resize(im, (im_w, im_h))
        lb = cv2.resize(lb, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        if (im_h, im_w) == (crop_h, crop_w): 
            return im, lb
        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            lb = np.pad(lb, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)

        im_h, im_w, _ = im.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        return dict(
            im=im[sh:sh+crop_h, sw:sw+crop_w, :].copy(),
            lb=lb[sh:sh+crop_h, sw:sw+crop_w].copy()
        )
    
class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im, lb):
        if np.random.random() < self.p:
            return im, lb
        assert im.shape[:2] == lb.shape[:2]
        return im[:, ::-1, :], lb[:, ::-1]
    
class ColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness >= 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast >= 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation >= 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im, lb):
        assert im.shape[:2] == lb.shape[:2]
        if not self.brightness is None:
            rate = np.random.uniform(*self.brightness)
            im = self.adj_brightness(im, rate)
        if not self.contrast is None:
            rate = np.random.uniform(*self.contrast)
            im = self.adj_contrast(im, rate)
        if not self.saturation is None:
            rate = np.random.uniform(*self.saturation)
            im = self.adj_saturation(im, rate)
        return im, lb

    def adj_saturation(self, im, rate):
        M = np.float32([
            [1+2*rate, 1-rate, 1-rate],
            [1-rate, 1+2*rate, 1-rate],
            [1-rate, 1-rate, 1+2*rate]
        ])
        shape = im.shape
        im = np.matmul(im.reshape(-1, 3), M).reshape(shape)/3
        im = np.clip(im, 0, 255).astype(np.uint8)
        return im

    def adj_brightness(self, im, rate):
        table = np.array([
            i * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]

    def adj_contrast(self, im, rate):
        table = np.array([
            74 + (i - 74) * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]

class Compose(object):

    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im, lb):
        for comp in self.do_list:
            im, lb = comp(im, lb)
        return im, lb
    
# ========== ENET DATA AUGMENTATION ==========
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