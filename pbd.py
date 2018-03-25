#!/usr/bin/env python 

import numpy as np
import operator
import matplotlib.pyplot as plt
import logging

class Worker(object):
    def __init__(self, idx, obj, surrogate_obj, h, theta, pop_score, pop_params, use_logger, asynchronous=False):
        self.idx = idx
        
        self.use_logger = use_logger
        if use_logger:
            self.logger = logging.getLogger("Worker-{}".format(self.idx))
        else:
            print("Beginning Worker-{}".format(self.idx))
            
        self.asynchronous = asynchronous

        self.obj = obj
        self.surrogate_obj = surrogate_obj
        self.theta = theta
        self.h = h
        
        self.score = 0 # current score
        self.pop_score = pop_score # reference to population statistics
        self.pop_params = pop_params
        
        # for plotting
        self.theta_history = []
        self.Q_history = []
        
        self.rms = 0 # for rmsprop
        
        self.update() # intialize population
        
    def step(self, vanilla=False, rmsprop=False, Adam=False):
        """one step of GD"""
        decay_rate = 0.9
        alpha = 0.01
        eps = 1e-5
        
        d_surrogate_obj = -2.0 * self.h * self.theta
        
        if vanilla:
            self.theta += d_surrogate_obj * alpha # ascent to maximize function
        else:
            self.rms = decay_rate * self.rms + (1-decay_rate) * d_surrogate_obj**2
            self.theta += alpha * d_surrogate_obj / (np.sqrt(self.rms) + eps)
                                
    def eval(self):
        """metric we want to optimize e.g mean episodic return or validation set performance"""
        self.score = self.obj(self.theta)
        return self.score
        
    def exploit(self):
        """copy weights, hyperparams from the member in the population with the highest performance"""
        if self.asynchronous:
            pop_score, pop_params = self.proxy_sync(pull=True)
        else:
            pop_score = self.pop_score
            pop_params = self.pop_params
            
        best_worker_idx = max(pop_score.items(), key=operator.itemgetter(1))[0]
        if best_worker_idx != self.idx:
            # print(self.idx, pop_score) enable to check if shared memory is being updated
            
            best_worker_theta, best_worker_h = pop_params[best_worker_idx]
            self.theta = np.copy(best_worker_theta)
            
            if self.use_logger:
                self.logger.info("Inherited optimal weights from Worker-{}".format(best_worker_idx))
            else:
                print("Worker-{} Inherited optimal weights from Worker-{}".format(self.idx, best_worker_idx))
            return True
        return False
        
    def explore(self):
        """perturb hyperparameters with noise from a normal distribution"""
        eps = np.random.randn(*self.h.shape) * 0.1
        self.h += eps
        
    def update(self):
        """update worker stats in global dictionary"""
        if not self.asynchronous:
            self.pop_score[self.idx] = self.score
            self.pop_params[self.idx] = (np.copy(self.theta), np.copy(self.h)) # np arrays are mutable
        else:
            self.proxy_sync(push=True)
            
        self.theta_history.append(np.copy(self.theta))
        self.Q_history.append(self.score)
        
        if len(self.Q_history) % 10 == 0:
            if self.use_logger:
                self.logger.info("Q = {:0.2f} ({:0.2f}%)".format(self.score, self.score * 100 / 1.2))
            else:
                print("Worker-{} Step {} Q = {:0.2f} ({:0.2f}%)".format(
                                                            self.idx, 
                                                            len(self.Q_history),
                                                            self.score, 
                                                            self.score * 100 / 1.2),
                                                            )
                                                                                      
    def proxy_sync(self, pull=False, push=False):
        """for asynchronous workers, we need to sync the values to the shared proxies
        https://docs.python.org/2/library/multiprocessing.html#multiprocessing.managers.SyncManager.list
        """
        
        if pull: # grab newest copy of pop_params
            return self.pop_score[0], self.pop_params[0]

        if push: # update newest copy
            _pop_score = self.pop_score[0]
            _pop_params = self.pop_params[0]
            
            _pop_score[self.idx] = self.score
            _pop_params[self.idx] = (np.copy(self.theta), np.copy(self.h))
            
            self.pop_score[0] = _pop_score
            self.pop_params[0] = _pop_params

def run(steps=200, explore=True, exploit=True):
    # Q and Q_hat, as per fig. 2: https://arxiv.org/pdf/1711.09846.pdf
    obj = lambda theta: 1.2 - np.sum(theta**2)
    surrogate_obj = lambda theta, h: 1.2 - np.sum(h*theta**2)

    pop_score = {} # score for all members
    pop_params = {} # params for all members
        
    # initialize two workers 
    population = [
        Worker(1, obj, surrogate_obj, np.array([1.,0.]), np.array([0.9, 0.9]), pop_score, pop_params, True),
        Worker(2, obj, surrogate_obj, np.array([0.,1.]), np.array([0.9, 0.9]), pop_score, pop_params, True),
        ]
        
    for step in range(steps):
        for worker in population:
            
            worker.step(vanilla=True) # one step of GD
            worker.eval() # evaluate current model
            
            if step % 10 == 0:
                if explore and exploit:
                    do_explore = worker.exploit()                
                    if do_explore:
                        worker.explore()
                        
                elif explore and not exploit:
                    worker.explore()
                
                elif not explore and exploit:
                    worker.exploit()
                    
                elif not explore and not exploit:
                    pass
                    
            worker.update()

    return population
    
def plot_Q(run, i, steps, title):
    
    plt.subplot(2,4,i)
    plt.plot(run[0].Q_history, color='b', lw=0.7)
    plt.plot(run[1].Q_history, color='r', lw=0.7)
    plt.axhline(y=1.2, linestyle='dotted', color='k')
    axes = plt.gca()
    axes.set_xlim([0,steps])
    axes.set_ylim([0.0, 1.21])
    
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Q')
    
def plot_theta(run, i, steps, title):
    x_b = [_[0] for _ in run[0].theta_history]
    y_b = [_[1] for _ in run[0].theta_history]
    
    x_r = [_[0] for _ in run[1].theta_history]
    y_r = [_[1] for _ in run[1].theta_history]
    
    plt.subplot(2,4,i)
    plt.scatter(x_b, y_b, color='b', s=2)
    plt.scatter(x_r, y_r, color='r', s=2)
    
    plt.title(title)
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    
def main():
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d %(name)s %(message)s',
                        datefmt="%M:%S")
    
    steps = 200
    
    run1 = run(steps=steps)
    run2 = run(steps=steps, exploit=False)
    run3 = run(steps=steps, explore=False)
    run4 = run(steps=steps, exploit=False, explore=False)
    
    
    plot_Q(run1, 3, steps=steps, title='PBT')
    plot_Q(run2, 4, steps=steps, title='Explore only')
    plot_Q(run3, 7, steps=steps, title='Exploit only')
    plot_Q(run4, 8, steps=steps, title='Grid Search')
    
    plot_theta(run1, 1, steps=steps, title='PBT')
    plot_theta(run2, 2, steps=steps, title='Explore only')
    plot_theta(run3, 5, steps=steps, title='Exploit only')
    plot_theta(run4, 6, steps=steps, title='Grid Search')
    
    plt.show()
    
    
if __name__ == '__main__':
    main()
        
    
        