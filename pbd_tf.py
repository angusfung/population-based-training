import argparse
import sys

import tensorflow as tf

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                            job_name=FLAGS.job_name,
                            task_index=FLAGS.task_index)
                            
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
                
            # model
            
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                is_chief=(FLAGS.task_index == 0)) as mon_sess:
                
                for i in range(10):
                    # print(i, FLAGS.task_index)
                    _place, value = mon_sess.run([W, y])
                    print(value)

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

