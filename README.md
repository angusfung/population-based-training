# population-based-training

To reproduce the results from DeepMind's paper on [Population Based Training of Neural Networks](https://arxiv.org/pdf/1711.09846.pdf).

PBT is an optimization algorithm that maximizes the performance of a network by optimizating a population of models and their hyperparameters. It determines a *schedule* of hyperparameter settings using an evolutionary strategy of exploration and exploitation - a much more powerful method than simply using a fixed set of hyperparameters throughout the entire training or using grid-search which is time-extensive. 

### Toy Example
The toy example was reproduced from fig. 2 in the paper (pg. 6). The idea is to maximize an unknown quadratic equation `Q = 1.2 - w1^2 - w2^2`, given a surrogate function `Q_hat = 1.2 - h1 w1^2 - h2 w2^2`, where `h1` and `h2` are hyperparameters and `w1` and `w2` are weights. Training begins with a `Population`, consisting of a set of `Workers` each with their own weights and hyperparameters. During exploration, the hyperparameters are perturbed by gaussian noise, and during exploitation, a `Worker` inherits the weights of the best `Worker` in the population. As per the paper, only `two` workers were used. 

The reproduced plots are seen below:
![alt text](https://github.com/angusfung/population-based-training/blob/master/plots.png)
Some key observations: 
* Theta Plots
   * In *Exploit only*, the intersection of the workers represents the inheritance of best weights from one worker to the other; this occurs every `10` steps (set by the user)
   * In *Explore only*, we don't see any intersections. Each point follows closely from the last from random perturbations and gradient descent steps
   * In *PBT*, we see the combination of the aformentioned effects
* `Q` Plots
   * The *Grid search*, plot never converges to `1.2` due to bad initialization. As the hyperparameters are **fixed** during the entire training, `Worker1` with `h=[1 0]` and `Worker2` with `h=[0 1]`, the surrogate function will never converge to the real function with `h=[1 1]`. This illustrates the shortcomings of grid-search, which can limit the generalization capabilities of a model (especically with bad initializations).

#### Run
 `./pbd.py` or `./toy_example.py`
 `pbd.py` was the original implementation of the toy example, but much complexity has been added to it to support other scripts. For a clean implementation of the toy example, please read `toy_example.py`.
