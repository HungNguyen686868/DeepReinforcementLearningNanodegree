# Continous Control-Project

## Table of Contents

 * [Problem_statement](#problem-statement)
 * [Requirements](#requirements)
 * [File Descriptions](#file-descriptions)
 * [Acknowledgements](#acknowledgements)

## Problem statement
In this project, I trained an agent to maintain its position at the target location for as many time steps as possible.
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


## Requirements
This project should be run with these following libraries:
- numpy
- matplotlib
- torch >= 0.4.0
- python >= 3.6

This project environment is already built thank to Udacity, and you can download it from one of the links below. If you want to reproduce my results, please consider using the Udacity environment provided by Udacity [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).

## File Descriptions
This repository contains:
- A notebook that explain all codes in my project
- Report file that give details about my approach

## Acknowledgements
- Thanks Udacity for great project [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)
- [DDPG paper](https://arxiv.org/pdf/1509.02971v6.pdf)


