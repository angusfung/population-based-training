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
        
        # use replica_device_setter to automatically set device-ops
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
                
            """
            can't modify MutableHashTable once MTS finalizes the graph, 
            although a mapped assign might work
            """
            # num_workers = len(worker_hosts)
            # global_weights = tf.contrib.lookup.MutableHashTable(
            #                     key_dtype=tf.string, # worker idx (int doesn't work here)
            #                     value_dtype=tf.float32, # weights
            #                     default_value=-999,
            #                     )
            
            best_weights = tf.tuple(
                [
                    tf.get_variable(
                        name='worker_idx',
                        dtype=tf.int32, 
                        initializer=tf.constant(-1, dtype=tf.int32), 
                        trainable=False), 
                        
                    tf.get_variable(
                        name='weight',
                        dtype=tf.float64, 
                        initializer=tf.constant(np.array([-1.,-1.])), 
                        trainable=False),
                ])
        
            weight_placeholder = tf.tuple([tf.placeholder(dtype=tf.int32, shape=[1]), tf.placeholder(dtype=tf.float32, shape=[2])])
            assign_weights = tf.contrib.framework.nest.map_structure(
                                lambda state, var: tf.assign(var, state),
                                weight_placeholder,
                                best_weights,
                                check_types=False
                                )
                                                        
            # create model
            surrogate_obj = 1.2 - tf.reduce_sum(tf.multiply(h, tf.square(theta)))
            obj = 1.2 - tf.reduce_sum(tf.square(theta))
            
            loss = tf.square((obj - surrogate_obj))
            
            optimizer = tf.train.AdamOptimizer(1e-4)
            train_step = optimizer.minimize(loss)
            
            tf.summary.scalar('loss', loss)
            merged = tf.summary.merge_all()
            
            random_index = tf.constant(5)
            random_str = tf.as_string(random_index)
            
            
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

