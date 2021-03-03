## README 

"[clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://arxiv.org/abs/2003.07311)"

CVPR 2021

Authors:  Johannes C. Paetzold and Suprosanna Shit


## Table of contents

* [General info](#general-info)
* [Segmenting data](#test)
* [Training a model](#train)
* [Feature extraction](#feats)
* [Dependencies](#depend)

## General info

For each of the described tasks, segmentation, training and feature extraction we have  set up a working example (demo) in the code ocean compute capsule. 

This repository includes four main blocks:

1. A utility library for the Deep Learning part of the work in KERAS including 
	
	a. The complete deep learning library including all networks and utilities
	
	b. Detailed descriptions how to setup this framework (Readme.md).	
	
2.  A data folder containing:
	
	a. A part of the synthetic dataset used to pretrain our network	

	b. An exemplary testing/training dataset and corresponding ground truth annotations

	c. The whole data can be found here (http://discotechnologies.org/). 

3. A model folder containing:
	
	a. Models trained on sythetic data for 1 and 2 channel network input
	
	b. The fully refined 2 input channel network
	
4. MATLAB scripts for statistical evaluation of features
