import numpy as np

class Worker(object):
    def __init__(self, idx, obj, surrogate_obj, theta, h):
        self.idx = idx
        self.obj = obj
        self.surrogate_obj = surrogate_obj
        self.theta = theta
        self.h = h
        
        self.score = None # current score

    def step(self):
        """one step of SGD with RMSProp""":
        d_surrogate_obj = -2 * self.h * np.dot(self.h, self.theta)
        self.theta -= d_surrogate_obj * 0.001
                        
    def eval(self, population_score, population_params):
        """metric we want to optimize e.g mean episodic return or validation set performance"""
        self.score = self.obj(self.theta)
        self.update() # update worker stats
        return self.score
        
    def exploit(self):
        """copy weights from the member in the population with the highest performance"""
        
        
    def explore(self):
        """perturb hyperparaters with noise from a normal distribution"""
        pass
        
    def update(self, population_score, population_params):
        """update worker stats in global dictionary"""
        population_score[self.idx] = self.score
        population_params[self.idx] = np.copy(self.theta) # np arrays are mutable
        
def main():
    # Q and Q_hat, as per fig. 2: https://arxiv.org/pdf/1711.09846.pdf
    obj = lambda theta: 1.2 - theta**2
    surrogate_obj = lambda theta, h: 1.2 - np.dot(h, theta**2)

    population_score = {} # score for all members
    population_params = {} # params for all members
    stats = (population_score, population_params)
        
    # initialize two workers 
    population = [
        worker1 = Worker(idx=1, obj, surrogate_obj, theta=np.array([1,0]), h=np.array([0.9, 0.9])),
        worker2 = Worker(idx=2, obj, surrogate_obj, theta=np.array([0,1]), h=np,array([0.9, 0.9])),
        ]
        
    for step in range(200):
        for worker in population:
            
            worker.step() # one step of GD
            score = worker.eval() # evaluate current model
            
            if step % 5 == 0:
                worker.exploit()
                worker.explore()
                
                

if __name__ == '__main__':
    main()
        
    
        