from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from data_loader import DataLoader
from nets import *

class PixelCNN(object):
    def __init__(self):
        pass
    
    def build_train_graph(self):
        opt = self.opt
        
        loader = DataLoader(dataset_name=opt.dataset_name,
                            batch_size=opt.batch_size)
        
        with tf.name_scope("data_loading"):
            images = loader.X_ph
            labels = loader.y_ph
            output_dim = images.get_shape()[3].value

        with tf.name_scope("prediction"):
            pred = pixel_cnn(images,num_filters=opt.num_block_cnn_filters,num_layers=opt.num_block_cnn_layers,h=labels)

        with tf.name_scope("compute_loss"):
            per_image_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=images, logits=pred)
            total_loss = tf.reduce_mean(tf.reduce_sum(tf.layers.flatten(per_image_loss),axis=-1))

        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
            self.train_op = tf.contrib.training.create_train_op(total_loss, optim)
            self.global_step = tf.Variable(0, 
                                           name='global_step', 
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step, 
                                              self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.loader = loader
        self.pred = tf.nn.sigmoid(pred)
        self.total_loss = total_loss
        self.images = images
        self.labels = labels
    
    def collect_summaries(self):
        opt = self.opt
        
        tf.summary.scalar('loss', self.total_loss, collections=['train'], family='train')
        tf.summary.image('image', self.images, max_outputs=5, collections=['train'], family='train')
        tf.summary.image('pred', tf.clip_by_value(self.pred,0.,1.), max_outputs=5,collections=['train'], family='train')
        self.train_summary_op = tf.summary.merge_all('train')

        tf.summary.scalar('loss', self.total_loss, collections=['test'], family='test')
        tf.summary.image('image', self.images, max_outputs=5, collections=['test'], family='test')
        tf.summary.image('pred', tf.clip_by_value(self.pred,0.,1.), max_outputs=5, collections=['test'], family='test')
        self.test_summary_op = tf.summary.merge_all('test')

    def train(self, opt):
        self.opt = opt
        self.build_train_graph()
        self.collect_summaries()

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.trainable_variables()] + \
                                    [self.global_step],
                                     max_to_keep=10)
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                                 save_summaries_secs=0, 
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            for step in range(1, opt.max_steps):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }

                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["summary"] = self.train_summary_op
                
                X_train, y_train = self.loader.load_train_batch()
                feed_dict = {
                    self.loader.X_ph: X_train,
                    self.loader.y_ph: y_train
                }

                if step == 2:
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    results = sess.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(chrome_trace)
                else:
                    results = sess.run(fetches,feed_dict=feed_dict)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    test_fetches = {
                        "loss": self.total_loss,
                        "summary": self.test_summary_op
                    }
                    X_test, y_test = self.loader.load_test_batch()
                    test_feed_dict = {
                        self.loader.X_ph: X_test,
                        self.loader_y_ph: y_test
                    }
                    test_results = sess.run(test_fetches,feed_dict=test_feed_dict)

                    sv.summary_computed(sess,results["summary"], gs)
                    sv.summary_computed(sess,test_results["summary"], gs)

                    print("%6d time: %4.4f/it train loss: %.3f, test loss: %.3f" \
                            % (step, \
                                (time.time() - start_time)/opt.summary_freq, 
                                results["loss"],test_results["loss"]))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)

    def setup_inference(self,opt): 
        self.opt = opt
        self.build_test_graph()

    def build_test_graph(self):
        opt = self.opt

        image_ph = tf.placeholder('float32',(opt.batch_size,opt.image_height,opt.image_width,1))
        if opt.num_classes > 1:
            label_ph = tf.placeholder('float32',(opt.batch_size,opt.num_classes))
        else:
            label_ph = None

        with tf.name_scope("prediction"):
            pred = pixel_cnn(image_ph,opt.num_block_cnn_filters,opt.num_block_cnn_layers,h=label_ph)

        self.image_ph = image_ph
        self.label_ph = label_ph
        self.pred = pred

    def predict(self, image, label, sess):
        if self.label_ph is not None:
            results = sess.run(self.pred, feed_dict={self.image_ph:image,self.label_ph:label})
        else:
            results = sess.run(self.pred, feed_dict={self.image_ph:image})
        return results

