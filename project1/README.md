# DeepReinforcementLearningNanodegree

## Table of Contents

 * [Project Overview](#project-overview)
 * [Problem_statement](#problem-statement)
 * [Requirements](#requirements)
 * [File Descriptions](#file-descriptions)
 * [Dataset](#dataset)
 * [Medium Blog Post](#medium-blog-post)
 * [Acknowledgements](#acknowledgements)

## Problem statement
In this project, I trained an agent to navigate and collect bananas (yellow and blue) in a large, square world.
If an agent collects a yellow banana, it will get a reward of +1. Otherwise, it will get a reward of -1 for collecting a blue banana. Therefore, the goal of an agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.


## Requirements
This project should be run with these following libraries:
- numPy

This project environment is similar to, but not identical to the Banana Collector environment on the Unity ML-Agents GitHub page. So, my code might not work in this enviroment. If you want to reproduce my results, please consider using the environment provided by Udacity [here](https://github.com/udacity/Value-based-methods#dependencies).

## File Descriptions
