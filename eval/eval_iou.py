# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import os
import sys
import time
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize

from dataset import cityscapes
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

# Import networks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../train')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.erfnet import ERFNet
from train.enet import ENet
from train.bisenet import BiSeNet


NUM_CHANNELS = 3
NUM_CLASSES = 20

# Preprocessing
image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])

target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    if args.model == "erfnet":
      model = ERFNet(NUM_CLASSES).to(device)
    elif args.model == "erfnet_isomaxplus":
      model = ERFNet(NUM_CLASSES, use_isomaxplus=True).to(device)
      # Check if the model has loss_first_part in the decoder
      # print("loss_first_part present in the decoder:",
      #   hasattr(model.decoder if not isinstance(model, torch.nn.DataParallel) else model.module.decoder, 'loss_first_part'))
    elif args.model =="enet":
        model = ENet(NUM_CLASSES).to(device)
    elif args.model == "bisenet":
        model = BiSeNet(NUM_CLASSES).to(device)

    # model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()
    
    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()

        # Check if the model is ERFNet with IsoMaxPlusLossFirstPart
        if args.model == "erfnet_isomaxplus" and 'loss_first_part_state_dict' in state_dict:
            loss_first_part_state_dict = state_dict['loss_first_part_state_dict']
            if hasattr(model.module.decoder, 'loss_first_part'):
                model.module.decoder.loss_first_part.load_state_dict(loss_first_part_state_dict)
            else:
                raise ValueError("IsoMaxPlusLossFirstPart not found in the model")

        if 'state_dict' in state_dict:
            load_dict = state_dict['state_dict'] 
        else:
            load_dict = state_dict

        for name, param in load_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model
    
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if args.method == "void":
        iouEvalVal = iouEval(NUM_CLASSES, 20)
    else:
        iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        if args.model == "bisenet":
            result = outputs[0]
        else:
            result = outputs

        iouEvalVal.addBatch(result.max(1)[1].unsqueeze(1).data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 

        # print (step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    if args.method == "void":
        print(iou_classes_str[19], "void")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  # can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=2)   # to avoid UserWarning of excessive worker creation
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--model', default="erfnet") # can be erfnet, erfnet_isomaxplus, enet, bisenet
    parser.add_argument('--method', type=str, default=None)  # can be MSP, MaxLogit, MaxEntropy, void
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
