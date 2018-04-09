import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import time

tf.logging.set_verbosity(tf.logging.INFO)

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
            
            
            with tf.name_scope('global_variables'):
                global_weights = tf.contrib.lookup.MutableHashTable(
                                    key_dtype=tf.string,
                                    value_dtype=tf.float32,
                                    default_value=[999.,999.],
                                    )
                
                global_loss = tf.contrib.lookup.MutableHashTable(
                                    key_dtype=tf.string, 
                                    value_dtype=tf.float32,
                                    default_value=999.,
                                    )
                                
            with tf.name_scope('main_graph'):
                # create model
                surrogate_obj = 1.2 - tf.reduce_sum(tf.multiply(h, tf.square(theta)))
                obj = 1.2 - tf.reduce_sum(tf.square(theta))
                
                loss = tf.square((obj - surrogate_obj))
                
                optimizer = tf.train.AdamOptimizer(1e-1)
                train_step = optimizer.minimize(loss)
                
                tf.summary.histogram('theta', theta)
                tf.summary.scalar('surrogate_obj', surrogate_obj)
                tf.summary.scalar('loss', loss)
                merged = tf.summary.merge_all()
                
            with tf.name_scope('update_graph'):
                """update worker stats in population"""
                def update():
                    global_weights_ops = global_weights.insert(tf.constant(str(FLAGS.task_index)), theta)
                    global_loss_ops = global_loss.insert(tf.constant(str(FLAGS.task_index)), loss)
                    
                    return global_weights_ops, global_loss_ops
                    
                do_update = update()
                
            with tf.name_scope('exploit_graph'):
                """copy weights from the member in the population with the highest performance"""
                def find_best_worker_idx():
                    # initialize
                    worker_index_summation = tf.constant(0)
                    
                    best_loss = tf.constant(999.)
                    best_idx = tf.constant(-1)
                    
                    def cond(index, best_loss, best_idx):
                        return tf.less(index, len(worker_hosts))
                        
                    def body(index, best_loss, best_idx):
                        """
                        compares worker loss with population member loss (in a loop)
                        returns best loss
                        """
                        def update_best_loss():
                            return member_loss, index
                        
                        def keep_best_loss():
                            return best_loss, best_idx
                            
                        member_loss = global_loss.lookup(tf.as_string(index))
                        best_loss, best_idx = tf.cond(
                                        member_loss < best_loss,
                                        true_fn=update_best_loss,
                                        false_fn=keep_best_loss,
                                        )
                                        
                        return index+1, best_loss, best_idx
                    
                    return tf.while_loop(
                                    cond=cond, 
                                    body=body, 
                                    loop_vars=[worker_index_summation, best_loss, best_idx], 
                                    back_prop=False
                                    )
                    
                def exploit():
                    """returns a weight assign op"""
                    _, best_worker_loss, best_worker_idx = find_best_worker_idx()
                    
                    def inherit_weights():
                        _ = tf.Print(
                                input_=best_worker_idx,
                                data=[best_worker_idx], 
                                message="Inherited optimal weights from Worker-")
                                
                        best_worker_weights = global_weights.lookup(tf.as_string(best_worker_idx))
                        return _, theta.assign(best_worker_weights)
                        
                    def keep_weights():
                        _ = tf.Print(
                                input_=tf.constant(1),
                                data=[], 
                                message="Continue with current weights")
                                
                        return _, tf.identity(theta)
                    
                    _, theta_ops = tf.cond(
                                    tf.not_equal(best_worker_idx, tf.cast(worker_idx, tf.int32)),
                                    true_fn=inherit_weights,
                                    false_fn=keep_weights,
                                    )
                    # for debug
                    # return loss, best_worker_loss, best_worker_idx
                
                    return _, theta_ops
                    
                do_exploit = exploit()
                
            with tf.name_scope('explore_graph'):
                def explore():
                    return h.assign(h + tf.random_normal(shape=[2]) * 0.1)
                    
                do_explore = explore()
        
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                is_chief=True) as mon_sess:

                # create log writer object (log from each machine)
                writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
                for step in range(50):
                    
                    time.sleep(0.25) # small delay
                                    
                    summary, h_, theta_, loss_, _= mon_sess.run([merged, h, theta, loss, train_step])
                    print("Worker {}, Step {}, h = {}, theta = {}, loss = {:0.6f}".format(
                                                                                    FLAGS.task_index,
                                                                                    step,
                                                                                    h_,
                                                                                    theta_,
                                                                                    loss_
                                                                                    ))
                    writer.add_summary(summary, step)
                    
                    if step % 5 == 0:
                        mon_sess.run([do_exploit]) # exploit
                        mon_sess.run([do_explore]) # explore
 
                    mon_sess.run([do_update]) # update

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

