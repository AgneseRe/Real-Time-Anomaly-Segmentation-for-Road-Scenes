import cv2
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF

from dataset import cityscapes
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
    def __init__(self, crop_size=(512, 512), p_rotation=.5):
        self.crop_size = crop_size
        self.p_rotation = p_rotation
        self.scales = [0.75, 1.0, 1.5, 1.75, 2]

    def __call__(self, img, mask):
        # Random resize
        scale = random.uniform(*self.scales)
        img, mask = self.resize(img, mask, scale)

        # Random crop
        img, mask = self.random_crop(img, mask)

        # Random rotation
        if random.random() < self.p_rotation:
            angle = random.uniform(-15, 15)  # Rotate between -15 and 15 degrees
            img, mask = self.rotate(img, mask, angle)

        # Random horizontal flip
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            mask = mask[:, ::-1]

        # Normalize and convert to tensor
        img = self.to_tensor(img)
        mask = torch.tensor(mask, dtype=torch.long)

        return img, mask

    def resize(self, img, mask, scale):
        """ Resize the image and mask using a given scale. """
        new_h = int(img.shape[0] * scale)
        new_w = int(img.shape[1] * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return img_resized, mask_resized

    def random_crop(self, img, mask):
        """ Randomly crop the image and mask to the specified size. """
        h, w, _ = img.shape
        crop_h, crop_w = self.crop_size
        if h < crop_h or w < crop_w:
            # Pad if the image is smaller than the crop size
            pad_h = max(crop_h - h, 0)
            pad_w = max(crop_w - w, 0)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=255)

        h, w, _ = img.shape
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        img_cropped = img[top:top+crop_h, left:left+crop_w, :]
        mask_cropped = mask[top:top+crop_h, left:left+crop_w]
        return img_cropped, mask_cropped

    def rotate(self, img, mask, angle):
        """ Rotate the image and mask by a given angle. """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rotated = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_rotated = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        return img_rotated, mask_rotated

    def to_tensor(self, img):
        """ Convert image to tensor format. """
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # Normalize to [0, 1]
        return torch.tensor(img)
    
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