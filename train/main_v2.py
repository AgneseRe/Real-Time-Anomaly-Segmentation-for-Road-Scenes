import os
import sys
import math
import time
import torch
import random
import numpy as np

from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from dataset import VOC12, cityscapes
from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard

import importlib
from iouEval import iouEval, getColorEntry
from shutil import copyfile

# import augmentations transformations and loss functions
from utils.augmentations import ErfNetTransform, BiSeNetTransform, ENetTransform
from utils.losses.focal_loss import FocalLoss
from utils.losses.ohem_ce_loss import OhemCELoss
from utils.losses.ce_loss import CrossEntropyLoss2d
from utils.losses.logit_norm_loss import LogitNormLoss

# import functions for weights computation
from utils.weights import calculate_enet_weights, calculate_erfnet_weights

NUM_CHANNELS = 3
NUM_CLASSES = 20 # cityscapes dataset (19 + 1)

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

# Mean computation for data normalization ([0.2869, 0.3251, 0.2839])
def compute_cs_mean(dataset_path, num_workers, batch_size):
    """
    Compute per-channel pixel values mean (RGB) over all images in the train subset
    of cityscapes. The computed mean is used to normalize the data before the training.

    Parameters:
        - dataset_path (str): The path to the Cityscapes dataset.
        - num_workers (int): The number of workers to use for data loading.
        - batch_size (int): The batch size to use when loading the data.

    Returns:
        list: A list of three float values representing the mean for the R, G, B channels.
    """

    if os.path.exists("./utils/cityscapes_mean.npy"):
        return np.load("./utils/cityscapes_mean.npy").tolist()
    else:
        # if mean not yet computed
        dataset = cityscapes(dataset_path, co_transform=None, subset='train')
        loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
        
        total_images = 0  # 2795
        cs_mean = torch.zeros(3)

        for images, _ in loader:
            batch_images = images.size(0)  # (batch_size, channels, height, width)
            total_images += batch_images
            cs_mean += images.mean(dim=[0, 2, 3]) * batch_images
        cs_mean = cs_mean / total_images

        # Save mean values in file for future computation
        np.save("./utils/cityscapes_mean.npy", cs_mean.numpy())
      
        return cs_mean.tolist()
    

# ========== TRAIN FUNCTION ==========
def train(args, model, enc=False):
    """
    Train a deep learning model using the Cityscapes dataset.

    Parameters:
        - args (argparse.Namespace): Configuration object containing attributes for training
            the path of the dataset directory, the batch size for training and validation, 
            the number of epochs to train, whether to compute IoU during training etc.
        - model (torch.nn.Module): Pytorch model to train.
        - enc (boolean): Flag for indicating whether to train the encoder or the decoder part.
    """
    best_acc = 0

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded" 

    # Augmentations Transformations (different models)
    if args.model == "erfnet":
        co_transform = ErfNetTransform(enc, augment=True, height=args.height)
        co_transform_val = ErfNetTransform(enc, augment=False, height=args.height)
    elif args.model == "bisenet":
        # print(f"Mean: {compute_cs_mean(args.datadir, args.num_workers, args.batch_size)}")
        cs_mean = compute_cs_mean(args.datadir, args.num_workers, args.batch_size)
        co_transform = BiSeNetTransform(cs_mean)
        co_transform_val = BiSeNetTransform(cs_mean)
    else:   # ENet
        co_transform = ENetTransform(augment=True)
        co_transform_val = ENetTransform(augment=False)

    # Dataset and Loader (train and validation both)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # Calculate weights of the model (torch tensor)
    if args.model == "erfnet":
        weights = calculate_erfnet_weights(enc, NUM_CLASSES)
    elif args.model == "enet":
        if os.path.exists("./utils/enet_weights.npy"):
            weights = torch.tensor(np.load("./utils/enet_weights.npy"))
        else:
            weights = calculate_enet_weights(loader, NUM_CLASSES)
            np.save("./utils/enet_weights.npy", weights.numpy())    
    else:   # BiSeNet
        weights = None

    if weights is not None:
        if args.cuda:
            weights = weights.cuda()
            print(weights)
    
    # Loss function
    if args.model == "erfnet":
        if args.loss == "ce":   # Cross Entropy Loss
            criterion = CrossEntropyLoss2d(weights)
        elif args.loss == "focal":  # Focal Loss
            criterion = FocalLoss()
        
        if args.logit_norm: # Logit Normalization
            criterion = LogitNormLoss(loss_function = criterion)
        print(f"ErfNet criterion: {type(criterion)}")
    elif args.model == "enet":
        criterion = CrossEntropyLoss2d(weights)
    else:   # BiSeNet 
        # hard examples loss value greater than 0.7 by default
        criterion_principal = OhemCELoss()
        criterion_aux16 = OhemCELoss()
        criterion_aux32 = OhemCELoss()

    savedir = f'../save/{args.savedir}'

    # Encoder/Decoder part
    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    # do not add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    # TODO: finetuning

    # Optimizer
    # ========== ERFNET ==========
    # Official paper section IV Experiments: https://ieeexplore.ieee.org/document/8063438
    # GitHub: https://github.com/Eromera/erfnet/blob/master/train/opts.lua
    # ========== ENET ==========
    # Official paper section 5.2 Benchmarks: https://arxiv.org/abs/1606.02147
    # ========== BISENET ==========
    # Official paper section 4.1 Implementation Protocol: https://arxiv.org/abs/1808.00897
    if args.model == "erfnet":
        optimizer = Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=2e-4)
    elif args.model == "enet":
        optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=2e-4)
    else:   # BiSeNet
        optimizer = SGD(model.parameters(), lr=2.5e-2, momentum=0.9, weight_decay=1e-4)

    start_epoch = 1
    if args.resume:
        # Must load weights, optimizer, epoch and best value. 
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    # Learning Rate Scheduler
    if args.model == "erfnet" or args.model == "bisenet":
        lambda1 = lambda epoch: pow((1 - ((epoch-1)/args.num_epochs)), 0.9) 
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    else:   # ENet
        scheduler = lr_scheduler.StepLR(optimizer, 100, gamma = 0.1)

    # Model visualization
    if args.visualize and args.steps_plot > 0:
        board = Dashboard(args.port)

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)

        epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader):
            start_time = time.time()

            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            optimizer.zero_grad()

            if args.model == "erfnet":
                outputs = model(inputs, only_decode = enc)
            else:
                outputs = model(inputs)

            # compute loss (for BiSeNet combination of three components)
            if args.model == "bisenet":
                loss_principal = criterion_principal(outputs[0], targets[:, 0])
                loss_aux16 = criterion_aux16(outputs[1], targets[:, 0])
                loss_aux32 = criterion_aux32(outputs[2], targets[:, 0])
                loss = loss_principal + loss_aux16 + loss_aux32
            else:   # for ErfNet and ENet
                loss = criterion(outputs, targets[:, 0])

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                if args.model == "bisenet":
                    iouEvalTrain.addBatch(outputs[0].max(1)[1].unsqueeze(1).data, targets.data)
                else:
                    iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                
            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                
                board.image(image, f'input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):   # merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                    f'target (epoch: {epoch}, step: {step})')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        # Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        for step, (images, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images, volatile=True)    #volatile flag makes it free backward or outputs for eval
            targets = Variable(labels, volatile=True)
            
            if args.model == "erfnet":
                outputs = model(inputs, only_decode = enc)
            else:
                outputs = model(inputs)

            # compute loss (for BiSeNet combination of three components)
            if args.model == "bisenet":
                loss_principal = criterion_principal(outputs[0], targets[:, 0])
                loss_aux16 = criterion_aux16(outputs[1], targets[:, 0])
                loss_aux32 = criterion_aux32(outputs[2], targets[:, 0])
                loss = loss_principal + loss_aux16 + loss_aux32
            else:   # for ErfNet and ENet
                loss = criterion(outputs, targets[:, 0])

            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)


            # Add batch to calculate TP, FP and FN for iou estimation
            if (doIouVal):
                # start_time_iou = time.time()
                if args.model == "bisenet":
                    iouEvalVal.addBatch(outputs[0].max(1)[1].unsqueeze(1).data, targets.data)
                else:
                    iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                board.image(image, f'VAL input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):   # merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'VAL output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'VAL output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                    f'VAL target (epoch: {epoch}, step: {step})')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 
           

        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'    
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        # SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best.pth'
        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best.pth'

        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')

        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    return(model)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    # Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)

    if args.model == "erfnet":
        model = model_file.ErfNet(NUM_CLASSES)
    elif args.model == "bisenet":
        model = model_file.BiSeNet(NUM_CLASSES)
    else:   # ENet
        model = model_file.ENet(NUM_CLASSES)

    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")
    
    # weights for fine tuning 

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    
    if args.state:
        # if args.state is provided then load this state for training
        # Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        
        def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        #print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))


    #train(args, model)
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True) #Train encoder
    #CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0. 
    #We must reinit decoder weights or reload network passing only encoder in order to train decoder
    print("========== DECODER TRAINING ===========")
    if (not args.state):
        if args.pretrainedEncoder:
            print("Loading encoder pretrained in imagenet")
            from erfnet_imagenet import ERFNet as ERFNet_imagenet
            pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
            pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
            pretrainedEnc = next(pretrainedEnc.children()).features.encoder
            if (not args.cuda):
                pretrainedEnc = pretrainedEnc.cpu()     #because loaded encoder is probably saved in cuda
        else:
            pretrainedEnc = next(model.children()).encoder
        model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)  #Add decoder to encoder
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()
        #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model = train(args, model, False)   #Train decoder
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    # You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument('--resume', action='store_true')    # Use this flag to load last checkpoint for training  

    parser.add_argument('--loss', default='ce') # ["ce", "focal"]
    parser.add_argument('--logit_norm', action='store_true', default=False)
    parser.add_argument('--loadWeights', default='erfnet_pretrained.pth')

    main(parser.parse_args())
