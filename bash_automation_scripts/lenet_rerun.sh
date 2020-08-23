#!/bin/bash

echo "LENET SKETCHES "
for i in {1..5}
do 
    python train_finegrained.py --model lenet --resize True --category sketches
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


echo "LENET ICONS "
for i in {1..5}
do 
    python train_finegrained.py --model lenet --resize True --category icons
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


echo "LENET FLAGS "
for i in {1..5}
do 
    python train_finegrained.py --model lenet --resize True --category flags
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