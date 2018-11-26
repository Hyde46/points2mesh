import os
import argparse
import tensorflow as tf
import numpy as np
import cv2

from tensorpack import *
from layers import flex_convolution, flex_pooling, knn_bruteforce

enable_argscope_for_module(tf.payers)

TOTAL_BATCH_SIZE = 16
BATCH_SIZE = 16


PC = {'num': 1024, 'dp':3}

class Model(ModelDesc):
    def inputs(self):
        reuturn [tf.placeholder(tf.float32, (None, PC['dp'], PC['num']), 'positions']
                
    def build_graph(self, positions):
        print tf.shape(positions)
        features = positions
        neighbors = _knn_bruteforce(positions, K=8 )

        x = features

        def subsample(x):
            n = x.shape.as_list()[-1]
            return x[:, :, :n // 4]
        def upsample (x):
            """
            Super simple upsampling
            """
            return tf.concat([x,x],0) 

        # Encoder
        for stage in range(3):
            x = flex_convolution(x, positions, neighbors, 64 / ((1 +  stage ) * 2), activation=tf.nn.relu)
            x = flex_pooling(x, neighbors)
            x = subsample(x)
            positions = subsample(positions)
            neighbors = knn_bruteforce(positions, K=8 )

        # Decoder
        for stage in range(3):
            x = flex_convolution(x, positions, neighbors, 4 * ((1 + stage) * 2), activation=tf.nn.relu)
            x = flex_pooling(x, neighbors)
            x = upsample(x)
            positions = upsample(positions)
            neighbors = knn_bruteforce(positions, K= 8)
        r = flex_convolution(x, positions, neighbors, 1, activation=tf.nn.sigmoid)
        print tf.shape(r)
        # axis ? 
        total_cost = tf.reduce_sum(tf.math.subtract(positions,r,name="difference"))
        return total_cost


    def optimizer(self):
        return tf.train.AdamOptimizer(1e-4)


def get_data():
    df_train = []

    df_test = []
    return df_train, df_test


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
  parser.add_argument('--load', help='load model')
  parser.add_argument('--fusion', help='run sampling', default='',
                      choices=['pooling', 'conv'])
  args = parser.parse_args()

  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  logger.set_logger_dir('train_log/fusion_%s' % (args.fusion))

  dataset_train, dataset_test = get_data()
  steps_per_epoch = len(dataset_train)

  config = TrainConfig(
        model=Model(),
        data=FeedInput(dataset_train),
        steps_per_epoch=steps_per_epoch,
        max_epoch=100
          )
  launch_train_with_config(config, SimpleTrainer())
