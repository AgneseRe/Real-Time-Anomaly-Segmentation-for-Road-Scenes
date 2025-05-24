import os
import sys
import math
import time
import torch
import random
import importlib
import numpy as np

from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, lr_scheduler
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from dataset import VOC12, cityscapes
from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard

from iouEval import iouEval, getColorEntry
from shutil import copyfile

# Import loss functions
from utils.losses.focal_loss import FocalLoss
from utils.losses.ohem_ce_loss import OhemCELoss
from utils.losses.combined_loss import CombinedLoss
from utils.losses.ce_loss import CrossEntropyLoss2d
from utils.losses.logit_norm_loss import LogitNormLoss
from utils.losses.isomax_plus_loss import IsoMaxPlusLossVanilla

# Import functions for class weights computation and data augmentation
from utils.weights import calculate_enet_weights, calculate_erfnet_weights, calculate_erfnet_weights_hard
from utils.augmentations import ErfNetTransform, BiSeNetTransform, ENetTransform

NUM_CHANNELS = 3
NUM_CLASSES = 20    # Cityscapes dataset (19 + 1)

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

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

    # ========== DATA AUGMENTATION ==========
    if args.model == "erfnet" or args.model == "erfnet_isomaxplus":
        co_transform = ErfNetTransform(enc, augment=True, height=args.height)
        co_transform_val = ErfNetTransform(enc, augment=False, height=args.height)
    elif args.model == "enet":
        co_transform = ENetTransform(augment=True, height=args.height)
        co_transform_val = ENetTransform(augment=False, height=args.height)
    else:   # BiSeNet
        co_transform = BiSeNetTransform(augment=True)
        co_transform_val = BiSeNetTransform(augment=False)

    # ========== TRAIN AND VAL DATASET ==========
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # ========== CLASS WEIGHTS ==========
    if args.model == "erfnet" or args.model == "erfnet_isomaxplus":
        if args.class_weights == "hard":
            weights = calculate_erfnet_weights_hard(enc, NUM_CLASSES)
        else:   # by processing dataset histogram
            mode = "encoder" if enc else "decoder"
            if not os.path.exists(f"./utils/class_distribution/erfnet_class_weights_{mode}.npy"):
                print("Calculating class weights...")
                weights = calculate_erfnet_weights(loader, NUM_CLASSES, enc)
            else:
                weights = torch.tensor(np.load(f"./utils/class_distribution/erfnet_class_weights_{mode}.npy"))
    elif args.model == "enet":
        if not os.path.exists("./utils/class_distribution/enet_class_weights.npy"):
            print("Calculating class weights...")
            weights = calculate_enet_weights(loader, NUM_CLASSES)
        else:
            weights = torch.tensor(np.load("./utils/class_distribution/enet_class_weights.npy"))   
    else: 
        weights = None  # BiSeNet learns from unbalanced dataset 

    if weights is not None:
        if args.cuda:
            weights = weights.cuda()
            # print(weights)

    # ========== LOSS FUNCTION ==========
    def get_base_loss(args, weights=None):
        """
        Get the base loss function for training based on provided arguments.

        Parameters:
            - args (argparse.Namespace): Configuration object containing attributes for training.
                It must contain the loss function to use and the logit normalization flag.
            - weights (torch.Tensor, optional): Class weights for the loss function. Default is None.

        Returns:
            - nn.Module: A loss function module based on the specified configuration.
        """
        if args.loss == "focal":    # Focal Loss
            base_loss = FocalLoss(gamma=2.0, alpha=weights)
        elif args.loss == "ce":     # Cross Entropy Loss
            base_loss = CrossEntropyLoss2d(weights)
        else:
            raise ValueError(f"Unsupported loss function: {args.loss}")
        
        # CE + Logit Normalization or Focal + Logit Normalization
        if args.logit_norm:
            base_loss = LogitNormLoss(loss=base_loss)

        return base_loss
    
    # Set the loss function based on the model and arguments
    if args.model == "erfnet":
        if args.loss == "ce":
            criterion = CrossEntropyLoss2d(weights)
        elif args.loss == "f":
            criterion = FocalLoss(gamma=2.0, alpha=weights)
        elif args.loss == "eim":
            criterion = IsoMaxPlusLossVanilla(entropic_scale=10.0)
        elif args.loss == "cef":
            criterion = CombinedLoss(
                ce_loss=CrossEntropyLoss2d(weights),
                focal_loss=FocalLoss(gamma=2.0, alpha=weights),
                alpha=1/2, beta=1/2, gamma=0.0)
        elif args.loss == "ceim":
            criterion = CombinedLoss(
                ce_loss=CrossEntropyLoss2d(weights),
                eim_loss=IsoMaxPlusLossVanilla(entropic_scale=10.0),
                alpha=1/2, beta=0.0, gamma=1/2)
        elif args.loss == "cefeim":
            criterion = CombinedLoss(
                ce_loss=CrossEntropyLoss2d(weights),
                focal_loss=FocalLoss(gamma=2.0, alpha=weights),
                eim_loss=IsoMaxPlusLossVanilla(entropic_scale=10.0),
                alpha=1/3, beta=1/3, gamma=1/3)
        else:
            raise ValueError(f"Unsupported loss function: {args.loss}")
        if args.logit_norm: # Logit Normalization
            criterion = LogitNormLoss(loss=criterion)
    elif args.model == "enet":
        criterion = CrossEntropyLoss2d(weights)
    else:   # BiSeNet hard examples loss value greater than 0.7 by default
        criterion_principal = OhemCELoss()
        criterion_aux16 = OhemCELoss()
        criterion_aux32 = OhemCELoss()

    print(f"Criterion: {type(criterion_principal) if args.model == 'bisenet' else type(criterion)}")

    savedir = f'../save/{args.savedir}'

    # ========== ENCODER/DECODER ==========
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

    # ========== FINE-TUNING ========== 
    if args.FineTune:
        # freezing all layers except the last one
        for param in model.parameters():
            param.requires_grad = False
        
        if args.model == "erfnet" or args.model == "erfnet_isomaxplus":
            for param in model.module.decoder.output_conv.parameters():
                param.requires_grad = True
        elif args.model == "enet":
            for param in model.module.transposed_conv.parameters():
                param.requires_grad = True
        else: # BiSeNet
            for param in model.module.conv_out.parameters():
                param.requires_grad = True
        
    # ========== OPTIMIZER ==========
    # References for optimizer configurations used in different segmentation models.

    # ERFNet:
    # - Paper Section IV Experiments: https://ieeexplore.ieee.org/document/8063438
    # - GitHub: https://github.com/Eromera/erfnet/blob/master/train/opts.lua

    # ENet:
    # - Paper Section 5.2 Benchmarks: https://arxiv.org/abs/1606.02147

    # BiSeNet:
    # - Paper Section 4.1 Implementation Protocol: https://arxiv.org/abs/1808.00897
    if args.model == "erfnet" or args.model == "erfnet_isomaxplus":
        optimizer = Adam(model.parameters(), 5e-5 if args.FineTune else 5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    elif args.model == "enet":
        optimizer = Adam(model.parameters(), lr=5e-5 if args.FineTune else 5e-4, weight_decay=0.0002)
    else:   # BiSeNet
        optimizer = SGD(model.parameters(), lr=2.5e-3 if args.FineTune else 2.5e-2, momentum=0.9, weight_decay=1e-4)

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

        state_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))
            

    # ========== LEARNING RATE SCHEDULER ==========
    if args.model == "erfnet" or args.model == "erfnet_isomaxplus" or args.model == "bisenet":
        lambda1 = lambda epoch: pow((1 - ((epoch-1)/args.num_epochs)), 0.9) 
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    else:   # ENet
        scheduler = lr_scheduler.StepLR(optimizer, 7 if args.FineTune else 100, 0.1)

    if args.resume:
        # step the scheduler up to the epoch we are resuming from
        for _ in range(start_epoch-1):
            scheduler.step()

    # ========== MODEL VISUALIZATION ==========
    if args.visualize and args.steps_plot > 0:
        board = Dashboard(args.port)

    for epoch in range(start_epoch, args.num_epochs+1):
        # stop training if stop_epoch is reached (for elegant resume in notebook)
        if epoch > args.stop_epoch:
            break
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step()    

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

            if args.model == "erfnet" or args.model == "erfnet_isomaxplus":
                outputs = model(inputs, only_encode = enc)
            else:
                outputs = model(inputs)

            # compute loss (for BiSeNet combination of three components)
            if args.model == "bisenet":
                loss_principal = criterion_principal(outputs[0], targets[:, 0])
                loss_aux16 = criterion_aux16(outputs[1], targets[:, 0])
                loss_aux32 = criterion_aux32(outputs[2], targets[:, 0])
                loss = loss_principal + 0.4 * loss_aux16 + 0.4 * loss_aux32
            else:   # for ERFNet (also IsoMaxPlus) and ENet
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

            inputs = Variable(images, volatile=True)    # volatile flag makes it free backward or outputs for eval
            targets = Variable(labels, volatile=True)
            
            if args.model == "erfnet" or args.model == "erfnet_isomaxplus":
                outputs = model(inputs, only_encode = enc)
            else:
                outputs = model(inputs)

            # compute loss (for BiSeNet combination of three components)
            if args.model == "bisenet":
                loss_principal = criterion_principal(outputs[0], targets[:, 0])
                loss_aux16 = criterion_aux16(outputs[1], targets[:, 0])
                loss_aux32 = criterion_aux32(outputs[2], targets[:, 0])
                loss = loss_principal + 0.4 * loss_aux16 + 0.4 * loss_aux32
            else:   # for ERFNet and ENet
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

        
        if args.model == "efrnet_isomaxplus":
            # only for saving also the loss first part dict
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': str(model),
                'state_dict': model.state_dict(),
                'loss_first_part_state_dict': model.module.decoder.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filenameCheckpoint, filenameBest)
        else:
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

        def save_model(model, filename, save_isomax=False):
            state = {'state_dict': model.state_dict()}
            if save_isomax and hasattr(model.module.decoder, 'loss_first_part'):
                state['loss_first_part_state_dict'] = model.module.decoder.loss_first_part.state_dict()
            torch.save(state, filename)
        
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            if args.model == "erfnet_isomaxplus":
                save_model(model, filename, save_isomax=True)
            else:
                save_model(model, filename)
            print(f'save: {filename} (epoch: {epoch})')

        if (is_best):
            if args.model == "erfnet_isomaxplus":
                save_model(model, filenamebest, save_isomax=True)
            else:
                save_model(model, filenamebest)
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

def ensemble_inference(args):

    def load_model(model_name, class_name, weight_path):
        model_file = importlib.import_module(model_name)
        model = getattr(model_file, class_name)(NUM_CLASSES)
        checkpoint = torch.load(weight_path, map_location="cuda" if args.cuda else "cpu", weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state, strict=False)
        model = torch.nn.DataParallel(model)
        if args.cuda:
            model = model.cuda()
        model.eval()
        return model

    print("Loading ensemble models...")
    erfnet = load_model("erfnet", "ERFNet", "../save/erfnet_training_void/model_best.pth")
    enet = load_model("enet", "ENet", "../save/enet_training_void/model_best.pth")
    bisenet = load_model("bisenet", "BiSeNet", "../save/bisenet_training_void/model_best.pth")

    # Use BiSeNet transforms for consistent size
    co_transform_val = BiSeNetTransform(augment=False)
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    iouEval_ensemble = iouEval(NUM_CLASSES)

    print("Running ensemble inference...")
    with torch.no_grad():
        for images, labels in loader_val:
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            out_erfnet = erfnet(images)
            out_enet = enet(images)
            out_bisenet = bisenet(images)[0]  # first output is main one

            # Convert to probabilities
            p_erfnet = torch.softmax(out_erfnet, dim=1)
            p_enet = torch.softmax(out_enet, dim=1)
            p_bisenet = torch.softmax(out_bisenet, dim=1)

            # Average predictions (soft voting)
            avg_probs = (p_erfnet + p_enet + p_bisenet) / 3.0
            pred = avg_probs.argmax(dim=1, keepdim=True)

            iouEval_ensemble.addBatch(pred.data, labels.data)

        iouVal, iou_classes = iouEval_ensemble.getIoU()

        class_names = [
            "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic Light",
            "Traffic Sign", "Vegetation", "Terrain", "Sky", "Person", "Rider", "Car",
            "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
        ]

        iou_classes_str = []
        for i in range(iou_classes.size(0)):
            iouStr = getColorEntry(iou_classes[i]) + '{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
            iou_classes_str.append(iouStr)

        print("=======================================")
        print("Per-Class IoU:")
        for idx, name in enumerate(class_names):
            print(f"{iou_classes_str[idx]} {name}")
        print("=======================================")
        meanIoUStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal*100) + '\033[0m'
        print("Ensemble MEAN IoU:", meanIoUStr, "%")


def main(args):

    # ============ MODEL ENSEMBLE ============
    if args.ensemble:
        print("========== ENSEMBLE INFERENCE ===========")
        ensemble_inference(args)
        return
    
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    # Load Model
    if args.model == "erfnet_isomaxplus":
        model_file = importlib.import_module("erfnet")
    else:
        assert os.path.exists(args.model + ".py"), "Error: model definition NOT FOUND"
        model_file = importlib.import_module(args.model)

    if args.model == "erfnet":
        model = model_file.ERFNet(NUM_CLASSES)
    elif args.model == "erfnet_isomaxplus":
        model = model_file.ERFNet(NUM_CLASSES, use_isomaxplus=True)
    elif args.model == "bisenet":
        model = model_file.BiSeNet(NUM_CLASSES)
    else:   # ENet
        model = model_file.ENet(NUM_CLASSES)

    # copyfile(args.model + ".py", savedir + '/' + args.model + ".py")
    
    # weights for fine tuning 
    if args.FineTune:
        weightspath =f"../trained_models/{args.loadWeights}"
        def load_my_state_dict(model, state_dict):
            own_state = model.state_dict()
            for name, param in state_dict.items():
                stripped_name = name[7:] if name.startswith("module.") else name
                if stripped_name in own_state:
                    if own_state[stripped_name].size() == param.size():
                        own_state[stripped_name].copy_(param)
                    elif "conv_out" in stripped_name:
                        # handles mismatches dimension conv_out
                        new_param = torch.zeros_like(own_state[stripped_name])
                        new_param[:param.size(0)] = param
                        own_state[stripped_name].copy_(new_param)
                    else:
                        print(f"Size mismatch for {stripped_name}: {own_state[stripped_name].size()} vs {param.size()}")
                else:
                    print(f"Skipping {name} as {stripped_name} is not in the model's state dict")
            return model
        if args.model == "enet":
          model = load_my_state_dict(model, torch.load(weightspath, map_location="cpu", weights_only=True)["state_dict"])
        else:
          model = load_my_state_dict(model, torch.load(weightspath, map_location="cpu", weights_only=True))
        print(f"Import Model {args.model} with weights {args.loadWeights} to FineTune")


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


    print("========== TRAINING ===========")
    if args.model == "erfnet" or args.model == 'erfnet_isomaxplus':
        if not args.FineTune:
            if (not args.decoder):
                print("========== ENCODER TRAINING ===========")
                model = train(args, model, True) #Train encoder
            # CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0. 
            # We must reinit decoder weights or reload network passing only encoder in order to train decoder
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
                model = model_file.ERFNet(NUM_CLASSES, encoder=pretrainedEnc)  #Add decoder to encoder
                if args.cuda:
                    model = torch.nn.DataParallel(model).cuda()
                # When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
        model = train(args, model, False)   #Train decoder
    elif args.model == "bisenet" or args.model == "enet":
        model = train(args,model)   
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
    parser.add_argument('--stop-epoch', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=2)   # 4
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

    parser.add_argument('--loss', default='ce') # 'ce', 'f', 'eim', 'cef', 'ceim', 'cefeim'
    parser.add_argument('--logit_norm', action='store_true', default=False) # Logit normalization
    parser.add_argument('--FineTune', action='store_true', default=False)
    parser.add_argument('--loadWeights', default='erfnet_pretrained.pth')
    parser.add_argument('--class-weights', default='hard') # Use hard weights or calculating by hist for ERFNet
    parser.add_argument('--ensemble', action='store_true', default=False, help="Run ensemble inference only")

    main(parser.parse_args())