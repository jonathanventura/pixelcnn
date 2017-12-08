from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from PixelCNN import PixelCNN
import os

flags = tf.app.flags
flags.DEFINE_string("dataset_name", "mnist", "Dataset name (mnist)")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_integer("batch_size", 128, "The size of a sample batch")
flags.DEFINE_integer("num_block_cnn_filters", 16, "Number of channels in BlockCNN filters")
flags.DEFINE_integer("num_block_cnn_layers", 7, "Number of BlockCNN layers")
flags.DEFINE_integer("max_steps", 200000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 1000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
FLAGS = flags.FLAGS

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
        
    seg = PixelCNN()
    seg.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
