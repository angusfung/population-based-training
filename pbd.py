import numpy as np

class Worker(object):
    def __init__(self, obj, surrogate_obj, theta, h):
        self.obj = obj
        self.surrogate_obj = surrogate_obj
        self.theta = theta
        self.h = h

    def step(self):
        """one step of SGD""":
        d_surrogate_obj = None
        
    
def main():
    # Q and Q_hat, as per fig. 2: https://arxiv.org/pdf/1711.09846.pdf
    obj = lambda theta: 1.2 - theta**2
    surrogate_obj = lambda theta, h: 1.2 - np.dot(h, theta**2)
    
    # initialize two workers 
    worker1 = Worker(obj, surrogate_obj, theta=np.array([1,0]), h=np.array([0.9, 0.9])
    worker2 = Worker(obj, surrogate_obj, theta=np.array([0,1]), h=np,array([0.9, 0.9])
    

if __name__ == '__main__':
    main()
        
    
        