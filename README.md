# population-based-training

To reproduce the results from DeepMind's paper on [Population Based Training of Neural Networks](https://arxiv.org/pdf/1711.09846.pdf). PBT is an optimization algorithm that maximizes the performance of a network by optimizating a population of models and their hyperparameters. It determines a *schedule* of hyperparameter settings using an evolutionary strategy of exploration and exploitation - a much more powerful method than simply using a fixed set of hyperparameters throughout the entire training. 

### Toy Example
The toy example was reproduced from fig. 2 in the paper (pg. 6). 

![alt text](https://github.com/angusfung/population-based-training/blob/master/plots.png)
