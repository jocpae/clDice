## README 

[clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://arxiv.org/abs/2003.07311)

CVPR 2021

Authors: Suprosanna Shit and Johannes C. Paetzold 


## Abstract 
Accurate segmentation of tubular, network-like structures, such as vessels, neurons, or roads, is relevant to many fields of research. For such structures, the topology is their most important characteristic; particularly preserving connectedness: in the case of vascular networks, missing a connected vessel entirely alters the blood-flow dynamics. We introduce a novel similarity measure termed centerlineDice (short clDice), which is calculated on the intersection of the segmentation masks and their (morphological) skeleta. We theoretically prove that clDice guarantees topology preservation up to homotopy equivalence for binary 2D and 3D segmentation. Extending this, we propose a computationally efficient, differentiable loss function (soft-clDice) for training arbitrary neural segmentation networks. We benchmark the soft-clDice loss on five public datasets, including vessels, roads and neurons (2D and 3D). Training on soft-clDice leads to segmentation with more accurate connectivity information, higher graph similarity, and better volumetric scores.


## Table of contents


* [clDice Metric](#metric)
* [clDice as a Loss function](#loss)
* [Dependencies](#depend)
* [Skeletonization](#skeleton)


## clDice Metric

## clDice as a Loss function

## Skeletonization

To use clDice as a loss function we introduce a differentiable soft-skeletonization where an iterative min- and max-pooling is applied as a proxy for morphological erosion and dilation.

![plot](https://github.com/jocpae/clDice/blob/main/skeletonization.png =250x)

