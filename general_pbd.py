#!/usr/bin/env python 
from pbd import Worker
from multiprocessing import Process, Manager

import operator
import time
import numpy as np
import matplotlib.pyplot as plt
import logging

# unfortunately multiprocessing module can't unpickle lambda functions
def obj(theta):
    return 1.2 - np.sum(theta**2)
    
def surrogate_obj(theta, h):
    return 1.2 - np.sum(h*theta**2)

def run(worker, steps, theta_dict, Q_dict):
    """start worker object asychronously"""
    for step in range(steps):
        worker.step(vanilla=True) # one step of GD
        worker.eval() # evaluate current model
        
        if step % 10 == 0:
            do_explore = worker.exploit()                
            if do_explore:
                worker.explore()
                                        
        worker.update()
    
    time.sleep(worker.idx) # to avoid race conditions
    
    _theta_dict = theta_dict[0]
    _Q_dict = Q_dict[0]
    _theta_dict[worker.idx] = worker.theta_history
    _Q_dict[worker.idx] = worker.Q_history
    theta_dict[0] = _theta_dict
    Q_dict[0] = _Q_dict
    
def plot(title, Q_history, steps, population_sizes):
    
    for population_size in population_sizes:
        plt.plot(Q_history[population_size], lw=0.7, label=str(population_size) + ' Worker')
    plt.axhline(y=1.2, linestyle='dotted', color='k')
    axes = plt.gca()
    axes.set_xlim([0,steps])
    axes.set_ylim([0.0, 1.21])
    
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Q')
    plt.legend(loc='upper right')
    
    plt.show()
        
def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d %(name)s %(message)s',
                        datefmt="%M:%S")

    Q_dict_with_size = {} # stores {population_size: Q_dict}
    theta_dict_with_size = {} # stores {population_size: theta}
    
    population_sizes = [1, 2, 4, 8, 16, 32]
    
    for population_size in population_sizes:        
        
        pop_score = Manager().list() # create a proxy for shared objects between processes
        pop_score.append({})
        
        pop_params = Manager().list()
        pop_params.append({})
        
        # population_size = 10
        steps = 50
        
        Population = [
                Worker(
                    idx=i, 
                    obj=obj, 
                    surrogate_obj=surrogate_obj, 
                    h=np.random.randn(2), 
                    theta=np.random.randn(2), 
                    pop_score=pop_score, 
                    pop_params=pop_params,
                    use_logger=False, # unfortunately difficult to use logger in multiprocessing
                    asynchronous=True, # enable shared memory between spawned processes
                    )
                    for i in range(population_size)
                    ]
        
        theta_dict = Manager().list()
        theta_dict.append({})
        Q_dict = Manager().list()
        Q_dict.append({})
        
        processes = []
        # create the processes to run asynchronously
        for worker in Population:
            _p = Process(
                    target=run, 
                    args=(worker,steps,theta_dict,Q_dict)
                    )
            processes.append(_p)
        
        # start the processes
        for i in range(population_size):
            processes[i].start()
        for i in range(population_size): # join to prevent Manager to shutdown
            processes[i].join()
    
        # find agent with best performance
        best_worker_idx = max(pop_score[0].items(), key=operator.itemgetter(1))[0]
        
        # save best agent/worker for a given population size
        Q_dict_with_size[population_size] = Q_dict[0][best_worker_idx]
        theta_dict_with_size[population_size] = theta_dict[0][best_worker_idx]
        
    # plot('Population of 10 Workers', Q_dict[0][best_worker_idx], steps
    plot('Q per step for various population sizes', Q_dict_with_size, steps, population_sizes)
    
    
    

if __name__ == '__main__':
    main()