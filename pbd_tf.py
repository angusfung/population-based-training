import argparse
import sys
import os

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
    logs_path = os.path.join(os.getcwd(), 'logs'.format(FLAGS.task_index))
                            
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
                
            # create model
            h = tf.get_variable('h', initializer=tf.random_uniform(shape=[2]), trainable=False)
            theta = tf.get_variable('theta', initializer=[0.9, 0.9])
            
            surrogate_obj = 1.2 - tf.reduce_sum(tf.multiply(h, tf.square(theta)))
            obj = 1.2 - tf.reduce_sum(tf.square(theta))
            
            loss = tf.square((obj - surrogate_obj))
            
            optimizer = tf.train.AdamOptimizer(1e-4)
            train_step = optimizer.minimize(loss)
            
            tf.summary.scalar('loss', loss)
            merged = tf.summary.merge_all()

            
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                is_chief=(FLAGS.task_index == 0)) as mon_sess:
                                                    
                # create log writer object (log from each machine)
                writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
                
                for i in range(1000):                    
                    summary, a, b, c, _= mon_sess.run([merged, h, theta, loss, train_step])
                    print(a, b, c)
                    writer.add_summary(summary, i)
                    

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

