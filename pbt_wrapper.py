import argparse

from multiprocessing import Process
from subprocess import Popen, PIPE


def create_worker(type, task, ps_hosts, worker_hosts, task_index):
    if task == "mueller":
        p = Popen([
            'python3', 'mueller_tf.py', ps_hosts, worker_hosts, '--job_name={}'.format(type), '--task_index={}'.format(task_index)])
    
    elif task == "toy":
        p = Popen([
            'python3', 'pbtv2_tf.py', ps_hosts, worker_hosts, '--job_name={}'.format(type), '--task_index={}'.format(task_index)])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int)
    parser.add_argument("task", type=str)
    args = parser.parse_args()
    
    population_size = args.size
    task = args.task
    
    # create cluster specifications
    ps_hosts = '--ps_hosts='
    worker_hosts = '--worker_hosts='
    
    hostnames = ['localhost:{}'.format(i) for i in range(2222, 2222+(args.size+1))]

    ps_hosts = ps_hosts + hostnames[0]
    worker_hosts = worker_hosts + ','.join(hostnames[1:])
    
    # create Processes
    processes = []
    
    
    for i in range(population_size):
        if i == 0:
            _p = Process(
                    target=create_worker, 
                    args=('ps', task, ps_hosts, worker_hosts, 0)
                    )
            processes.append(_p)
                    
        _p = Process(
                target=create_worker, 
                args=('worker', task, ps_hosts, worker_hosts, i)
                )
        processes.append(_p)


    for process in processes:
        process.start()
        
    for process in processes:
        process.join()
        
        
    
    
    
