# population-based-training

To reproduce the results from DeepMind's paper on [Population Based Training of Neural Networks](https://arxiv.org/pdf/1711.09846.pdf).

PBT is an optimization algorithm that maximizes the performance of a network by optimizating a population of models and their hyperparameters. It determines a *schedule* of hyperparameter settings using an evolutionary strategy of exploration and exploitation - a much more powerful method than simply using a fixed set of hyperparameters throughout the entire training or using grid-search which is time-extensive. 

### Toy Example
The toy example was reproduced from fig. 2 in the paper (pg. 6). The idea is to maximize an unknown quadratic equation `Q=1.2-w1^2-w2^2`, given a surrogate function `Q = 1.2 - h1 w1^2 - h2 w2^2`, where `h1` and `h2` are hyperparameters and `w1` and `w2` are weights. Training begins with a `Population`, which consists of a set of `Workers` each with their own weights and hyperparameters. 
During exploration, the hyperparameters are perturbed by gaussian noise, and during exploitation, 

![alt text](https://github.com/angusfung/population-based-training/blob/master/plots.png)
