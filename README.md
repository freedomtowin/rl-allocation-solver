# rl-allocation-solver

This post summarizes some business outcomes and goals that are considered while building mixed integer programming models in the allocation optimization space.

The developed source code creates the following: 1) a framework to simulate store-style-color target inventories, commited invetories, and bundles at the distribution center 2) a data pipeline to create an optimal MIP allocation proposals 3) and a reinforcement learning alogrithm to approximate allocation proposals.

It seems that the reinforcement learning algorithm was able to learn some strategy for allocating bundles to stores. However, the effectiveness of the algorithm is comparatively worse than the MIP solution and potentially greedy approaches for generating allocation proposals.

The purpose of this experiment was to see if a reinforcement learning (RL) algorithm could be used to approximate the allocation proposals of a mixed integer programming (MIP) model.

The use cases for the RL allocation algorithm are experimental, and improvements to the RL solutions and neural networks could be potentially change the use-case. It might be possible to use an RL algorithm to pre-solve some part of the MIP solution space. In some cases, it could be used in place of an MIP model. It might also be possible to build multiple RL models for different input data or scenarios.

The quality of the allocation proposals depends on more factors than on how close the solution was to the target inventory, i.e., it also depends on how much business value the model can generate. Here are a few factors to consider: