# Play Tennis Project

## Table of Contents

 * [Problem_statement](#problem-statement)
 * [Requirements](#requirements)
 * [File Descriptions](#file-descriptions)
 * [Acknowledgements](#acknowledgements)

## Problem statement
In this project, I trained 02 agent to maintain control rackets to bounce a ball over a net for as many time steps as possible.
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.



## Requirements
This project should be run with these following libraries:
- numpy
- matplotlib
- torch >= 0.4.0
- python >= 3.6

The project environment is similar to, but not identical to the Tennis environment on the Unity ML-Agents GitHub page. So, you can download it from one of the links below. If you want to reproduce my results, please consider using the Udacity environment provided by Udacity [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet). Then, you can train a agent by running a notebook file.

## File Descriptions
This repository contains:
- A notebook and its'html that explain all codes in my project
- Report file that give details about my approach
- Model checkpoint
- Learning Cuvre Figure

## Acknowledgements
- Thanks Udacity for great project 
- [TD3 paper](https://arxiv.org/abs/1802.09477v3)


