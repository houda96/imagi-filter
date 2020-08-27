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

def train(net, criterion, optimizer, transform, train_load, val_load, device, args):
    """ Train a network; reports train and test statistics after each epoch 
        Input: 
            - Network 
            - Loss function
            - Optimizer
            - Transformation on images
            - Train data in batches
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

    for epoch in tqdm(range(args.epochs), mininterval=10):
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        # Shuffle data
        train_load = [i for sub in train_load for i in sub]
        np.random.shuffle(train_load)
        train_load = [train_load[i:i + args.batch] for i in range(0, len(train_load), args.batch)]

        # Go over batches
        for batch_id, batch in enumerate(train_load):
            inputs = torch.stack(tuple([load_image(i[0], args.width, args.height, args.resize, transform) for i in batch]), 0).to(device)
            targets = torch.LongTensor([i[1] for i in batch]).to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Train accuracy calculations
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

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

        torch.save(net.state_dict(), args.model_save + args.model + "_test_" + str(epoch))

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

def plot_learning(model, epochs, train_acc, test_acc, train_losses, test_losses):
    plt.plot(list(range(epochs)), train_acc)
    plt.plot(list(range(epochs)), test_acc)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("accuracies1000_" + model + ".png")
    plt.show()

    plt.plot(list(range(epochs)), train_losses)
    plt.plot(list(range(epochs)), test_losses)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("losses1000_" + model + ".png")
    plt.show()

if __name__ == "__main__":
    # Arguments to give to use
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type = float, default = 0.0001, help = "learning rate, default=0.0001")
    parser.add_argument("--width", type = int, default = 300, help = "width images")
    parser.add_argument("--height", type = int, default = 300, help = "height images")
    parser.add_argument("--batch", type = int, default = 40, help = "batch size")
    parser.add_argument("--epochs", type = int, default = 30, help = "epochs")
    parser.add_argument("--pos_file", type = str, default = 'data/positive_examples/*')
    parser.add_argument("--neg_file", type = str, default = 'data/negative_examples/*')
    parser.add_argument("--folder_ims", type = str, default = 'data/')
    parser.add_argument("--resize", type = bool, default = False)
    parser.add_argument("--file_data", type = bool, default = True)
    parser.add_argument("--partialfile_t", type = str, default = "data/train_test_splits/train_load_3000.json")
    parser.add_argument("--partialfile_v", type = str, default = "data/train_test_splits/val_load_3000.json")
    parser.add_argument("--model_save", type = str, default = "trained_models/")
    parser.add_argument("--model", type = str, default = "lenet", choices=["lenet", "resnet", "vgg"],
            help = "chosen model, lenet, resnet, vgg")


    args = parser.parse_args()

    # Get by default device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

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
    print(sum(p.numel() for p in net.parameters()))
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    #train_load, val_load = get_data(args.pos_file, args.neg_file, args.batch, 0.9) # Uncomment to obtain slices
    if args.file_data:
        with open(args.partialfile_t, "r") as f:
            train_load = json.loads(f.read())
            train_load = [[[args.folder_ims + '/'.join(j[0].split('/')[-2:]), j[1]] for j in i] for i in train_load]

        with open(args.partialfile_v, "r") as f:
            val_load = json.loads(f.read())
            val_load = [[[args.folder_ims + '/'.join(j[0].split('/')[-2:]), j[1]] for j in i] for i in val_load]
    else:
        train_load, val_load = get_data(args.pos_file, args.neg_file, args.batch, 0.9)

    

    transform = TRANSFORMS[MODELS[args.model]]
    train_losses, train_acc, test_losses, test_acc = train(net, criterion, optimizer, transform,
                                                            train_load, val_load, device, args)
    print(train_losses)
    print(train_acc)
    print(test_losses)
    print(test_acc)
    plot_learning(args.model, args.epochs, train_acc, test_acc, train_losses, test_losses)
