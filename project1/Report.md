# Banana Project

## Introduction

## Solutions
1.
2.
3. DQNetwork
The DQNetwork is implemented in Pytorch. This network containts 04 fully connected layers (each layer have 64 units) and Relu activation function.
The hyperparameters are choosen from many attempts of training sessions on the CPU. These hyperparameters are given as follows:
* lr = 0.0001, gamma = 0.99, 
* epsilon = 1.0, 
* eps_dec = 1e-5, 
* eps_min = 0.01, 
* buffer_size = 10000, 
* batch_size = 32, 
* update = 4, 
* tau = 0.01
