#!/bin/bash

echo "VGG MAPS"
for i in 1 2 
do 
    python train_finegrained.py --model vgg --category maps
    echo ""
done
echo ""
echo ""

echo "RESNET MAPS "
for i in {1..5}
do 
    python train_finegrained.py --model resnet --category maps
    echo ""
done
echo ""
echo ""

echo "LENET MAPS "
for i in {1..5}
do 
    python train_finegrained.py --model lenet --resize True --category maps
    echo ""
done
echo ""
echo ""

echo "VGG ICONS"
for i in {1..5}
do 
    python train_finegrained.py --model vgg --category icons
    echo ""
done
echo ""
echo ""

echo "RESNET ICONS"
for i in {1..5}
do 
    python train_finegrained.py --model resnet --category icons
    echo ""
done
echo ""
echo ""

echo "LENET ICONS "
for i in {1..5}
do 
    python train_finegrained.py --model lenet --resize True --category icons
    echo ""
done
echo ""
echo ""

echo "VGG GRAPHS"
for i in {1..5}
do 
    python train_finegrained.py --model vgg --category graphs
    echo ""
done
echo ""
echo ""

echo "RESNET graphs"
for i in {1..5}
do 
    python train_finegrained.py --model resnet --category graphs
    echo ""
done
echo ""
echo ""

echo "LENET GRAPHS "
for i in {1..5}
do 
    python train_finegrained.py --model lenet --resize True --category graphs
    echo ""
done
echo ""
echo ""

echo "VGG FLAGS"
for i in {1..5}
do 
    python train_finegrained.py --model vgg --category flags
    echo ""
done
echo ""
echo ""

echo "RESNET FLAGS"
for i in {1..5}
do 
    python train_finegrained.py --model resnet --category flags
    echo ""
done
echo ""
echo ""

echo "LENET FLAGS "
for i in {1..5}
do 
    python train_finegrained.py --model lenet --resize True --category flags
    echo ""
done
echo ""
echo ""

echo "VGG BLACK"
for i in {1..5}
do 
    python train_finegrained.py --model vgg --category black
    echo ""
done
echo ""
echo ""

echo "RESNET BLACK"
for i in {1..5}
do 
    python train_finegrained.py --model resnet --category black
    echo ""
done
echo ""
echo ""

echo "LENET BLACK "
for i in {1..5}
do 
    python train_finegrained.py --model lenet --resize True --category black
    echo ""
done
echo ""
echo ""