import numpy as np
import operator
import matplotlib.pyplot as plt

class Worker(object):
    def __init__(self, idx, obj, surrogate_obj, theta, h, pop_score, pop_params):
        self.idx = idx
        self.obj = obj
        self.surrogate_obj = surrogate_obj
        self.theta = theta
        self.h = h
        
        self.score = 0 # current score
        self.pop_score = pop_score # reference to population statistics
        self.pop_params = pop_params
        self.loss_history = []
        
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
        self.val_performance = self.score / 1.2
        return self.score
        
    def exploit(self):
        """copy weights, hyperparams from the member in the population with the highest performance"""
        best_worker_idx = max(self.pop_score.items(), key=operator.itemgetter(1))[0]
        if best_worker_idx != self.idx:
            best_worker_theta, best_worker_h = self.pop_params[best_worker_idx]
            self.theta = np.copy(best_worker_theta)
            self.h = np.copy(best_worker_h)
            return True
        return False
        
    def explore(self):
        """perturb hyperparaters with noise from a normal distribution"""
        eps = np.random.randn(*self.h.shape) * 0.1
        self.h += eps
        
    def update(self):
        """update worker stats in global dictionary"""
        self.pop_score[self.idx] = self.score
        self.pop_params[self.idx] = (np.copy(self.theta), self.h) # np arrays are mutable
        self.loss_history.append(self.score)
        
def main():
    # Q and Q_hat, as per fig. 2: https://arxiv.org/pdf/1711.09846.pdf
    obj = lambda theta: 1.2 - np.sum(theta**2)
    surrogate_obj = lambda theta, h: 1.2 - np.sum(h*theta**2)

    pop_score = {} # score for all members
    pop_params = {} # params for all members
        
    # initialize two workers 
    population = [
        Worker(1, obj, surrogate_obj, np.array([1.,0.]), np.array([0.9, 0.9]), pop_score, pop_params),
        Worker(2, obj, surrogate_obj, np.array([0.,1.]), np.array([0.9, 0.9]), pop_score, pop_params),
        ]
    
    for step in range(100):
        for worker in population:
            
            worker.step(vanilla=True) # one step of GD
            worker.eval() # evaluate current model
            
            if step % 5 == 0:
                do_explore = worker.exploit()                
                if do_explore:
                    worker.explore()
                    
            worker.update()
            
    # print(population[0].loss_history[480:])
    # plt.subplot(2,2,1)
    # plt.plot(population[0].loss_history[480:])
    # plt.plot(population[1].loss_history)
    # axes = plt.gca()
    # # axes.set_xlim([0,100])
    # # axes.set_ylim([-1.2,1.2])
    # plt.show()
    

    
    
if __name__ == '__main__':
    main()
        
    
        