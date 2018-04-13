import argparse
import sys
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

# plotting
def mueller(X,Y):
    A = [-200., -100., -170., 15.]
    a = [-1., -1., -6.5, 0.7]
    b = [0., 0., 11., 0.6]
    c = [-10., -10., -6.5, 0.7]
    
    X0 = [1., 0., -0.5, -1.]
    Y0 = [0., 0.5, 1.5, 1.]
    
    Z = 0
    
    for i in range(4):
        Z += A[i]*np.exp(a[i]*(X-X0[i])**2 + b[i]*(X-X0[i])*(Y-Y0[i]) + c[i]*(Y-Y0[i])**2)
    return Z
    
def plot(ax=None, minx=-1.5, maxx=1.2, miny=-0.2, maxy=2, weights_history=None):
    grid_width = max(maxx-minx, maxy-miny) / 200.0
    xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
    V = mueller(xx, yy)
    ax.contourf(xx, yy, V.clip(max=200), 50)
    
    X = [_[0] for _ in weights_history]
    Y = [_[1] for _ in weights_history]
    
    ax.scatter(X, Y, color='b', s=2)
    
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
        
        num_hyperparams = 16
            
        with tf.device("/job:worker/task:{}".format(FLAGS.task_index)):
            
            worker_idx = tf.constant(FLAGS.task_index, dtype=tf.float32)
            
            # weights X, Y
            W = tf.get_variable(
                    'W'.format(FLAGS.task_index), 
                    # values taken from https://arxiv.org/pdf/1611.07657.pdf
                    initializer=tf.random_uniform(shape=[2], minval=[-2.,-0.5], maxval=[1.,2.])) 
                    #initializer=tf.random_uniform(shape=[4], minval=[-2.,-0.5, -1., -1.], maxval=[1.,2., 1., 1.])) 
                    
            # hyperparameters schedules 
            h = tf.get_variable('h', initializer=tf.random_uniform(shape=[num_hyperparams]), trainable=False)
            alpha = tf.get_variable('alpha', initializer=1e-1, trainable=False) # learning rate
            
            score = tf.get_variable('score', initializer=999., trainable=False)
            
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
                                    
                global_hyperparams = tf.contrib.lookup.MutableHashTable(
                                    key_dtype=tf.string,
                                    value_dtype=tf.float32,
                                    default_value=[999.,] * num_hyperparams
                                    )
                                    
                global_alpha = tf.contrib.lookup.MutableHashTable(
                                    key_dtype=tf.string,
                                    value_dtype=tf.float32,
                                    default_value=1e-1
                                    )
                
                global_loss = tf.contrib.lookup.MutableHashTable(
                                    key_dtype=tf.string, 
                                    value_dtype=tf.float32,
                                    default_value=999.,
                                    )
                # validation or test set
                global_score = tf.contrib.lookup.MutableHashTable( 
                                    key_dtype=tf.string, 
                                    value_dtype=tf.float32,
                                    default_value=999.,
                                    )
                                
            with tf.name_scope('main_graph'):
                
                # define constants, no lists in tf :/
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
                    
                # model = tf.nn.relu(W[3]*tf.nn.relu(W[2]*tf.nn.relu(tf.reduce_sum(h*W[0:1])))) # can add reg
                # model = tf.nn.relu(tf.reduce_sum(h*W[0:1]))
                # model = tf.nn.relu(tf.nn.relu(h[0] * W[0]) * h[1] * W[1])
                # model = tf.nn.relu(tf.nn.relu(W[0]) * W[1])
                # model = tf.sigmoid(tf.sigmoid(W[0]) * W[1])
                # model = tf.nn.relu(tf.nn.relu(tf.nn.relu(h[0] * W[0]) * h[1] * W[1]) * W[2])
                
                model = \
                    h[0] * tf.exp(h[4] * tf.square((W[0]-x0_1)) + h[8] * (W[0]-x0_1) * (W[1]-y0_1) + h[12] * tf.square((W[1]-y0_1))) + \
                    h[1] * tf.exp(h[5] * tf.square((W[0]-x0_2)) + h[9] * (W[0]-x0_2) * (W[1]-y0_2) + h[13] * tf.square((W[1]-y0_2))) + \
                    h[2] * tf.exp(h[6] * tf.square((W[0]-x0_3)) + h[10] * (W[0]-x0_3) * (W[1]-y0_3) + h[14] * tf.square((W[1]-y0_3))) + \
                    h[3] * tf.exp(h[7] * tf.square((W[0]-x0_4)) + h[11] * (W[0]-x0_4) * (W[1]-y0_4) + h[15] * tf.square((W[1]-y0_4)))
                
                # mueller_constant = tf.stop_gradient(mueller_potential)
                # loss = tf.square((mueller_constant-model))
                
                # loss = tf.square((mueller_potential-model))
                # loss = mueller_potential
                loss = model
                
                optimizer = tf.train.AdamOptimizer(alpha)
                train_step = optimizer.minimize(loss)
                
                # tf.summary.histogram('W', W)
                tf.summary.scalar('model', model)
                tf.summary.scalar('meuller_potential', mueller_potential)
                tf.summary.scalar('loss', loss)
                
                merged = tf.summary.merge_all()
                
            with tf.name_scope('update_graph'):
                """update worker stats in population"""
                def update():
                    global_weights_ops = global_weights.insert(tf.constant(str(FLAGS.task_index)), W)
                    global_hyperparams_ops = global_hyperparams.insert(tf.constant(str(FLAGS.task_index)), h)
                    global_loss_ops = global_loss.insert(tf.constant(str(FLAGS.task_index)), loss)
                    global_score_ops = global_score.insert(tf.constant(str(FLAGS.task_index)), score)
                    
                    return global_weights_ops, global_hyperparams_ops, global_loss_ops, global_score_ops
                    
                do_update = update()
                
            with tf.name_scope('exploit_graph'):
                """copy weights from the member in the population with the highest performance (based on score, not loss)"""
                def find_best_worker_idx():
                    # initialize
                    worker_index_summation = tf.constant(0)
                    
                    best_score = tf.constant(1e100)
                    best_idx = tf.constant(-1)
                    
                    def cond(index, best_score, best_idx):
                        return tf.less(index, len(worker_hosts))
                        
                    def body(index, best_score, best_idx):
                        """
                        compares worker score with population member score (in a loop)
                        returns best score
                        """
                        def update_best_score():
                            return member_score, index
                        
                        def keep_best_score():
                            return best_score, best_idx
                            
                        member_score = global_score.lookup(tf.as_string(index))
                        best_score, best_idx = tf.cond(
                                        member_score < best_score,
                                        true_fn=update_best_score,
                                        false_fn=keep_best_score,
                                        )
                                        
                        return index+1, best_score, best_idx
                    
                    return tf.while_loop(
                                    cond=cond, 
                                    body=body, 
                                    loop_vars=[worker_index_summation, best_score, best_idx], 
                                    back_prop=False
                                    )
                    
                def exploit():
                    """returns a weight and hyperparams assign op"""
                    _, best_worker_score, best_worker_idx = find_best_worker_idx()
                                        
                    def inherit_weights_hyperparams():
                        _ = tf.Print(
                                input_=best_worker_idx,
                                data=[best_worker_idx], 
                                message="Inherited optimal weights and hyperparams from Worker-")
                                
                        best_worker_weights = global_weights.lookup(tf.as_string(best_worker_idx))
                        best_worker_hyperparams = global_hyperparams.lookup(tf.as_string(best_worker_idx))
                        best_worker_alpha = global_alpha.lookup(tf.as_string(best_worker_idx))
                        return _, W.assign(best_worker_weights), h.assign(best_worker_hyperparams), \
                            alpha.assign(best_worker_alpha), tf.constant(1)
                    
                    def keep_weights():
                        _ = tf.Print(
                                input_=tf.constant(1),
                                data=[], 
                                message="Continue with current weights")
                                
                        return _, tf.identity(W), tf.identity(h), tf.identity(alpha), tf.constant(0) 
                    
                    _, W_ops, h_ops, alpha_ops, explore_flag = tf.cond(
                                                tf.not_equal(best_worker_idx, tf.cast(worker_idx, tf.int32)),
                                                true_fn=inherit_weights_hyperparams,
                                                false_fn=keep_weights,
                                                )
                    # for debug
                    # return loss, best_worker_loss, best_worker_idx, explore_flag
                    # return _, W_ops, h_ops, explore_flag, score, best_worker_score, best_worker_idx
                    
                    return _, W_ops, h_ops, alpha_ops, explore_flag
                    
                do_exploit = exploit()
                
            with tf.name_scope('explore_graph'):
                def explore():
                    h_ops = h.assign(h + tf.random_normal(shape=[num_hyperparams]) * 0.01)
                    
                    # 1.2 or 0.8 with equal probability
                    p = tf.random_uniform(shape=[], minval=0, maxval=1, dtype=tf.int32)
                    p_float = tf.cast(p, tf.float32)
                    
                    alpha_ops = alpha.assign(alpha * p_float* 1.2 + alpha * (1-p_float) * 0.8)
                    return h_ops, alpha_ops
                    
                do_explore = explore()
                
            with tf.name_scope('eval_graph'):
                # evaluate current model e.g test or validation set
                
                def eval():
                    return score.assign(mueller_potential)
                do_eval = eval()
                
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                is_chief=True) as mon_sess:

                # create log writer object (log from each machine)
                writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
                
                weights_history = []
                
                for step in range(200):
                    
                    time.sleep(0.25) # small delay
                                    
                    summary, h_, alpha_, W_, loss_, _ = mon_sess.run([merged, h, alpha, W, loss, train_step]) # step
                    score_ = mon_sess.run([do_eval]) # eval
                    
                    # note: does updating P make sense here? step could potentially 
                    # lead to a viable theta in which case we dont want to exploit
                    
                    mon_sess.run([do_update]) # update
                    
                    print("Worker {}, Step {}, h = {}, alpha = {}, W = {}, loss = {:0.6f}, score = {:0.6f}".format(
                                                                                                FLAGS.task_index,
                                                                                                step,
                                                                                                h_,
                                                                                                alpha_,
                                                                                                W_,
                                                                                                loss_,
                                                                                                score_[0],
                                                                                                ))
                    writer.add_summary(summary, step)
    
                    if step % 5 == 0:
                        _ = mon_sess.run([do_exploit]) # exploit
                        explore_flag = _[0][3]
                        
                        if explore_flag:
                            mon_sess.run([do_explore]) # explore
                        
                    mon_sess.run([do_update]) # update
                    
                    weights_history.append(W_)
                    
                    step += 1
            
            if FLAGS.task_index == 0: # arbitrary worker
                
                plot(ax=plt.gca(), weights_history=weights_history)
                plt.show()
            

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

