# Regular packages 
from PIL import Image
import numpy as np
import glob, os
import argparse
from load_data import load_image, get_data
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import json

# Torch 
from torchvision import transforms
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Models 
from models.lenet import LeNet
from models.resnet import ResNet152
from models.vgg import VGG19_BN

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

MODELS = {"lenet": 1, "vgg": 2, "resnet": 3}
TRANSFORMS = {1: transforms.Compose([transforms.ToTensor(), normalize]),
              2: transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize]),
              3: transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize])}

def train(net, criterion, optimizer, transform, train_load_ones, train_load_zeros, val_load, device, args):
    """ Train a network; reports train and test statistics after each epoch 
        Input: 
            - Network 
            - Loss function
            - Optimizer
            - Transformation on images
            - Train data in batches one label 
            - Train data in batches zero label
            - Validation data in batches 
            - Device to run on 
            - Passed arguments  

        Out (length is amount of epochs):
            - Losses on train data 
            - Accuracies on train data
            - Losses on test data 
            - Accuracies on test data
    """
    train_losses, train_accuracies, test_losses, test_accuracies = [], [], [], []
    number_of_updates = 0
    len_batches_zeros = len(train_load_zeros)  
    len_batches_ones = len(train_load_ones)
    idx_zero = 0
    idx_one = 0
    train_loss = 0
    correct = 0
    total = 0

    for update_i in tqdm(range(5000), mininterval=10):
        net.train()

        batch = train_load_zeros[idx_zero] + train_load_ones[idx_one]
        np.random.shuffle(batch)
        inputs = torch.stack(tuple([load_image(i[0], args.width, args.height, args.resize, transform) for i in batch]), 0).to(device)
        targets = torch.LongTensor([i[1] for i in batch]).to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        idx_zero += 1
        idx_one += 1

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if idx_zero == len_batches_zeros:
            idx_zero = 0
            train_load_zeros = [i for sub in train_load_zeros for i in sub]
            np.random.shuffle(train_load_zeros)
            train_load_zeros = [train_load_zeros[i:i + int(args.batch/2)] for i in range(0, len(train_load_zeros), int(args.batch/2))]

        if idx_one == len_batches_ones:
            idx_one = 0
            train_load_ones = [i for sub in train_load_ones for i in sub]
            np.random.shuffle(train_load_ones)
            train_load_ones = [train_load_ones[i:i + int(args.batch/2)] for i in range(0, len(train_load_ones), int(args.batch/2))]


        if update_i % args.step_eval == 0:
            # Report on train
            print("loss", "Acc")
            print(train_loss, correct/total*100)
            train_losses.append(train_loss)
            train_accuracies.append(correct/total*100)
            

            # Evaluate
            net.eval()
            val_load = [i for sub in val_load for i in sub]
            np.random.shuffle(val_load)
            val_load = [val_load[i:i + args.batch] for i in range(0, len(val_load), args.batch)]
            test_loss, correct, total = evaluate(net, val_load, transform, device, criterion, args)

            # Report on validation
            print("loss", "Acc")
            print(test_loss, correct/total*100)
            test_losses.append(test_loss)
            test_accuracies.append(correct/total*100)

            train_loss = 0
            correct = 0
            total = 0

            torch.save(net.state_dict(), args.model_save + args.category + '/' + args.model + "_test_" + str(update_i) + '.pt')

    return train_losses, train_accuracies, test_losses, test_accuracies

def evaluate(net, val_load, transform, device, criterion, args):
    """ Evaluate the network that is trained
        Input: 
            - Network 
            - Validation data in batches
            - Desired transformation on the images 
            - Which device to run on
            - Loss function
            - Args from main 
        Out:
            - Loss on validation 
            - Correct predictions
            - Total predictions
    """
    test_loss = 0
    correct = 0
    total = 0

    for batch_id, batch in enumerate(val_load):
        inputs = torch.stack(tuple([load_image(i[0], args.width, args.height, args.resize, transform) for i in batch]), 0).to(device)
        targets = torch.LongTensor([i[1] for i in batch]).to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        # Calculate correct predictions
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return test_loss, correct, total

if __name__ == "__main__":
    # Arguments to give to use
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type = float, default = 0.0001, help = "learning rate, default=0.0001")
    parser.add_argument("--width", type = int, default = 300, help = "width images")
    parser.add_argument("--height", type = int, default = 300, help = "height images")
    parser.add_argument("--batch", type = int, default = 40, help = "batch size")
    parser.add_argument("--step_eval", type=int, default=100, help="after how many updates, do eval")
    parser.add_argument("--model", type = str, default = "lenet", help = "chosen model, lenet, resnet, vgg")
    parser.add_argument("--folder_ims", type = str, default = 'ImageFilteringData/')
    parser.add_argument("--resize", type = bool, default = False)
    parser.add_argument("--file_data", type = bool, default = True)
    parser.add_argument("--category", type = str, default = "sketches", help="sketches, maps, icons, graphs, flags, black") # What category
    parser.add_argument("--model_save", type = str, default = "trained_models_finegrained/")
    parser.add_argument("--split_folder", type = str, default = "ImageFilteringData/splits_finegrained/")


    args = parser.parse_args()

    # Get by default device to use
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #print(device)

    # Load in chosen model
    if args.model == "lenet":
        net = LeNet(args.width).to(device)
    elif args.model == "resnet":
        net = ResNet152().to(device)
    elif args.model == "vgg":
        net = VGG19_BN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    print(args.model)
    # print(sum(p.numel() for p in net.parameters()))
    # print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    with open(args.split_folder + args.category + '_train_ones.json', 'r') as f:
        train_ones = json.load(f)

    with open(args.split_folder + args.category + '_train_zeros.json', 'r') as f:
        train_zeros = json.load(f)

    with open(args.split_folder + args.category + '_val_ones.json', 'r') as f:
        val_ones = json.load(f)

    with open(args.split_folder + args.category + '_val_zeros.json', 'r') as f:
        val_zeros = json.load(f)

    half_batch = int(args.batch/2)
    val_load = val_ones + val_zeros
    np.random.shuffle(val_load)
    val_load = [val_load[i:i + args.batch] for i in range(0, len(val_load), args.batch)]
    train_load_ones = [train_ones[i:i + half_batch] for i in range(0, len(train_ones), half_batch)]
    train_load_zeros = [train_zeros[i:i + half_batch] for i in range(0, len(train_zeros), half_batch)]

    

    transform = TRANSFORMS[MODELS[args.model]]
    train_losses, train_acc, test_losses, test_acc = train(net, criterion, optimizer, transform,
                                                            train_load_ones, train_load_zeros, val_load, device, args)
    print(train_losses)
    print(train_acc)
    print(test_losses)
    print(test_acc)
    print(max(test_acc), np.argmax(test_acc))
