import random
from PIL import Image, ImageOps
import numpy as np
import torch

from transform import Relabel, ToLabel
import torchvision.transforms.functional as TF
from torchvision.transforms import Pad, RandomCrop
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor


#Augmentations - different function implemented to perform random augments on both image and target
class ErfNetTransform(object):
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
    def __init__(self, height=512, width=1024, 
        mean=np.array([0.28689554, 0.32513303, 0.28389177]), 
        std=np.array([0.18696375, 0.19017339, 0.18720214]), 
        mode='train'):
        self.mode = mode
        self.height = height
        self.width = width
        self.scales = (0.75, 1.0, 1.5, 1.75, 2.0)
        self.mean = mean
        self.std = std

    def __call__(self, input, target):
        input = self.preprocess_image(input, Image.BILINEAR)
        target = self.preprocess_image(target, Image.NEAREST)

        if self.mode == 'train':
            input, target = self.augment_images(input, target)

        target = self.label_transform(target)

        return input, target

    def preprocess_image(self, image, interpolation):
        image = Resize(self.height, interpolation, antialias=True)(image)
        image = np.asarray(image).astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return torch.from_numpy(image.transpose(2, 0, 1)).float()

    def augment_images(self, input, target):
        if random.random() > 0.5:
            input = TF.hflip(input)
            target = TF.hflip(target)
        scale = random.choice(self.scales)
        size = int(scale * self.height)
        input = Resize(size, Image.BILINEAR, antialias=True)(input)
        target = Resize(size, Image.NEAREST, antialias=True)(target)
        if scale == 0.75:
            padding = (128, 256)
            input = Pad(padding, fill=0, padding_mode='constant')(input)
            target = Pad(padding, fill=255, padding_mode='constant')(target)
        i, j, h, w = RandomCrop.get_params(input, output_size=(self.height, self.width))
        input = TF.crop(input, i, j, h, w)
        target = TF.crop(target, i, j, h, w)
        return input, target

    def label_transform(self, target):
        target = ToLabel()(target)
        return Relabel(255, 19)(target)