import numpy as np
import operator

class Worker(object):
    def __init__(self, idx, obj, surrogate_obj, theta, h, pop_score, pop_params):
        self.idx = idx
        self.obj = obj
        self.surrogate_obj = surrogate_obj
        self.theta = theta
        self.h = h
        
        self.score = None # current score
        self.pop_score = pop_score # reference to population statistics
        self.pop_params = pop_params
        self.rms = 0 # for rmsprop

    def step(self):
        """one step of SGD with RMSProp"""
        decay_rate = 0.9
        alpha = 0.001
        
        d_surrogate_obj = -2.0 * self.h * self.theta
        self.rms = decay_rate * self.rms + (1-decay_rate) * d_surrogate_obj**2
        
    
        self.theta -= d_surrogate_obj * alpha
                        
    def eval(self):
        """metric we want to optimize e.g mean episodic return or validation set performance"""
        self.score = self.obj(self.theta)
        self.update() # update worker stats
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
        
    for step in range(200):
        for worker in population:
            
            worker.step() # one step of GD
            worker.eval() # evaluate current model
            
            if step % 5 == 0:
                do_explore = worker.exploit()                
                if do_explore:
                    worker.explore()
                    
            worker.update()
    print(pop_score)
    print(pop_params)
    
    
if __name__ == '__main__':
    main()
        
    
        