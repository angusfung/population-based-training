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
            # weights X, Y
            W = tf.get_variable(
                    'W'.format(FLAGS.task_index), 
                    # values taken from https://arxiv.org/pdf/1611.07657.pdf
                    initializer=tf.random_uniform(shape=[2], minval=[-2.,-0.5], maxval=[1.,2.])) 
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
                
                # define constants
                A_1 = tf.constant(-200.)
                A_2 = tf.constant(-100.)
                A_3 = tf.constant(-170.)
                A_4 = tf.constant(15.)
                
                a_1 = tf.constant(-1.)
                a_2 = tf.constant(-1.)
                a_3 = tf.constant(-6.5)
                a_4 = tf.constant(0.7)
                
                b_1 = tf.constant(0.)
                b_2 = tf.constant(0.)
                b_3 = tf.constant(11.)
                b_4 = tf.constant(0.6)
                
                c_1 = tf.constant(-10.)
                c_2 = tf.constant(-10.)
                c_3 = tf.constant(-6.5)
                c_4 = tf.constant(0.7)
                
                x0_1 = tf.constant(1.)
                x0_2 = tf.constant(0.)
                x0_3 = tf.constant(-0.5)
                x0_4 = tf.constant(-1.)
                
                y0_1 = tf.constant(0.)
                y0_2 = tf.constant(0.5)
                y0_3 = tf.constant(1.5)
                y0_4 = tf.constant(1.)
                
                mueller_potential = \
                    A_1 * tf.exp(a_1 * tf.square((W[0]-x0_1)) + b_1 * (W[0]-x0_1) * (W[1]-y0_1) + c_1 * tf.square((W[1]-y0_1))) + \
                    A_2 * tf.exp(a_2 * tf.square((W[0]-x0_2)) + b_2 * (W[0]-x0_2) * (W[1]-y0_2) + c_2 * tf.square((W[1]-y0_2))) + \
                    A_3 * tf.exp(a_3 * tf.square((W[0]-x0_3)) + b_3 * (W[0]-x0_3) * (W[1]-y0_3) + c_3 * tf.square((W[1]-y0_3))) + \
                    A_4 * tf.exp(a_4 * tf.square((W[0]-x0_4)) + b_4 * (W[0]-x0_4) * (W[1]-y0_4) + c_4 * tf.square((W[1]-y0_4)))
                    
                model = tf.nn.relu(tf.reduce_sum(h*W))
                
                loss = tf.square((mueller_potential-model))
                
                optimizer = tf.train.AdamOptimizer(1e-2)
                train_step = optimizer.minimize(loss)
                
                # tf.summary.histogram('theta', theta)
                # tf.summary.scalar('surrogate_obj', surrogate_obj)
                tf.summary.scalar('loss', loss)
                merged = tf.summary.merge_all()
                
            with tf.name_scope('update_graph'):
                """update worker stats in population"""
                def update():
                    global_weights_ops = global_weights.insert(tf.constant(str(FLAGS.task_index)), W)
                    global_loss_ops = global_loss.insert(tf.constant(str(FLAGS.task_index)), loss)
                    
                    return global_weights_ops, global_loss_ops
                    
                do_update = update()
                
            with tf.name_scope('exploit_graph'):
                """copy weights from the member in the population with the highest performance"""
                def find_best_worker_idx():
                    # initialize
                    worker_index_summation = tf.constant(0)
                    
                    best_loss = tf.constant(1e100)
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
                        return _, W.assign(best_worker_weights)
                        
                    def keep_weights():
                        _ = tf.Print(
                                input_=tf.constant(1),
                                data=[], 
                                message="Continue with current weights")
                                
                        return _, tf.identity(W)
                    
                    _, W_ops = tf.cond(
                                    tf.not_equal(best_worker_idx, tf.cast(worker_idx, tf.int32)),
                                    true_fn=inherit_weights,
                                    false_fn=keep_weights,
                                    )
                    # for debug
                    # return loss, best_worker_loss, best_worker_idx
                
                    return _, W_ops
                    
                do_exploit = exploit()
                
            with tf.name_scope('explore_graph'):
                def explore():
                    return h.assign(h + tf.random_normal(shape=[2]) * 0.01)
                    
                do_explore = explore()
        
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                is_chief=True) as mon_sess:

                # create log writer object (log from each machine)
                writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
                for step in range(50):
                    
                    time.sleep(0.25) # small delay
                                    
                    summary, h_, W_, loss_, _= mon_sess.run([merged, h, W, loss, train_step])
                    print("Worker {}, Step {}, h = {}, W = {}, loss = {:0.6f}".format(
                                                                                    FLAGS.task_index,
                                                                                    step,
                                                                                    h_,
                                                                                    W_,
                                                                                    loss_
                                                                                    ))
                    writer.add_summary(summary, step)
                
                    if step % 5 == 0:
                        mon_sess.run([do_exploit]) # exploit
                        mon_sess.run([do_explore]) # explore
 
                    mon_sess.run([do_update]) # update
                    
                    step += 1

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

