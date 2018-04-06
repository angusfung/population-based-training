import argparse
import sys
import os
import numpy as np
import tensorflow as tf

def main(_):
    # we need to provide all ps and worker info to each server so they are aware of each other
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    
    # create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    
    # create and start a server for the local task.
    server = tf.train.Server(cluster,
                            job_name=FLAGS.job_name,
                            task_index=FLAGS.task_index)
                            
    # log each worker seperately for tensorboard
    # https://github.com/tensorflow/tensorboard/blob/master/README.md#runs-comparing-different-executions-of-your-model
    logs_path = os.path.join(os.getcwd(), 'logs', '{}'.format(FLAGS.task_index))
                            
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        
        # explictely place weights and hyperparameters on the worker servers to prevent sharing
        # otherwise replica_device_setter will put them on the ps
        with tf.device("/job:worker/task:{}".format(FLAGS.task_index)):
            theta = tf.get_variable('theta'.format(FLAGS.task_index), initializer=tf.random_uniform(shape=[2]))
            h = tf.get_variable('h', initializer=tf.random_uniform(shape=[2]), trainable=False)
            worker_idx = tf.constant(FLAGS.task_index, dtype=tf.float32)
        
        # use replica_device_setter to automatically set device-ops
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
                
            #can't modify MutableHashTable once MTS finalizes the graph, 
            #although a mapped assign might work
            # num_workers = len(worker_hosts)
            # global_weights = tf.contrib.lookup.MutableHashTable(
            #                     key_dtype=tf.string, # worker idx (int doesn't work here)
            #                     value_dtype=tf.float32, # weights
            #                     default_value=-999,
            #                     )
            
            with tf.name_scope('global_variables'):
                best_worker_idx = tf.get_variable(
                                    name='best_idx', dtype=tf.float32, # must be float for tf.cond
                                    initializer=tf.constant(-1.), trainable=False)
                                
                idx_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
                best_worker_weight = tf.get_variable(
                                    name='best_weight',dtype=tf.float32,
                                    initializer=tf.constant([-1., -1.]), trainable=False)
                                
                best_worker_loss = tf.get_variable(
                                    name='best_loss', dtype=tf.float32,
                                    initializer=tf.constant(-999.), trainable=False)
            
            with tf.name_scope('main_graph'):
                # create model
                surrogate_obj = 1.2 - tf.reduce_sum(tf.multiply(h, tf.square(theta)))
                obj = 1.2 - tf.reduce_sum(tf.square(theta))
                
                loss = tf.square((obj - surrogate_obj))
                
                optimizer = tf.train.AdamOptimizer(1e-4)
                train_step = optimizer.minimize(loss)
                
                tf.summary.scalar('loss', loss)
                merged = tf.summary.merge_all()
            
            with tf.name_scope('exploit_graph'):
            # create mini graph for exploit updates
                def exploit(
                    worker_idx, worker_weight, worker_loss,
                    best_worker_idx, best_worker_weight, best_worker_loss,
                    ):
                    """
                    copy weights from the member in the population with the highest performance
                    
                    inputs:
                    -worker_idx:         rank 0 tensor (device index)
                    -worker_weight:      rank 1 tensor (weights)
                    -worker_loss:        ...
                    
                    -best_worker_idx:    rank 0 tensor (global best worker in population)
                    -best_worker_weight: rank 1 tensor (global best weights in population)
                    -best_worker_los     ...
                    
                    returns an assign op called update
                    """
                    
                    def push_weights():
                        print("Optimal weights found on Worker-{}".format(FLAGS.task_index)) 
                        update1 = best_worker_idx.assign(worker_idx)
                        # update2 = best_worker_weight.assign(worker_weight)
                        # update3 = best_worker_loss.assign(worker_loss)
                        
                        return update1
                        
                    def pull_weights():
                    
                        print("Inherited optimal weights from Worker-{}".format(FLAGS.task_index))
                        update = worker_weight.assign(best_worker_weight)
                        
                        return update
                    
                        
                    update = tf.cond(
                                    worker_loss > best_worker_loss, 
                                    true_fn=push_weights, 
                                    false_fn=pull_weights,
                                    strict=False,
                                    ) 
                    return update


                update_weights = exploit(
                                worker_idx, theta, loss, 
                                best_worker_idx, best_worker_weight, best_worker_loss)
            
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                is_chief=1) as mon_sess:

                
                # create log writer object (log from each machine)
                writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
                
                for step in range(200):                    
                    summary, h_, theta_, loss_, _= mon_sess.run([merged, h, theta, loss, train_step])
                    print("Worker {}, Step {}, h = {}, theta = {}, loss = {:0.3f}".format(
                                                                                    FLAGS.task_index,
                                                                                    step,
                                                                                    h_,
                                                                                    theta_,
                                                                                    loss_
                                                                                    ))
                    writer.add_summary(summary, step)
                    
                # print(mon_sess.run([best_worker_idx]))    
                # update = mon_sess.run([update_weights])
                # # mon_sess.run(update)
                # print(mon_sess.run(best_worker_idx))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )

    FLAGS, unparsed = parser.parse_known_args()
    
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

