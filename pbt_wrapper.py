from multiprocessing import Process
from subprocess import Popen, PIPE


def run():
    p = Popen(['python3', 'mueller_tf.py', '--ps_hosts=localhost:2222', '--worker_hosts=localhost:2223', '--job_name=ps', '--task_index=0'])
    p = Popen(['python3', 'mueller_tf.py', '--ps_hosts=localhost:2222', '--worker_hosts=localhost:2223', '--job_name=worker', '--task_index=0'])
    

if __name__ == "__main__":
    run()