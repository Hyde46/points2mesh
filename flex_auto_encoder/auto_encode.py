import os
import argparse
import tensorflow as tf
import numpy as np
import cv2

from tensorpack import *
from layers import (flex_convolution, flex_pooling, knn_bruteforce, flex_convolution_transpose)
from PointCloudDataFlow import *

enable_argscope_for_module(tf.layers)

TOTAL_BATCH_SIZE = 16
BATCH_SIZE = 64


PC = {'num': 1024, 'dp':3}



class Model(ModelDesc):

    def inputs(self):
        """
        Inputs are
        - pointcloud [dbatch,im_position, num_points]
        - Input should be equal to output if autoencoder is working nicely
        """
        #return [tf.placeholder(tf.float32, (PC['num'], PC['dp'], None), 'positions')]
        return [tf.placeholder(tf.float32, (None, PC['dp'], PC['num']), 'positions')]
                
    def _build_graph(self, positions):
        """
        Autoencoder
        """

        net = positions
        pos = positions
        neighbors = knn_bruteforce( pos, K=8 )

        def subsample(x):
            """
            Simple subsampling.
            """
            n = x.shape.as_list()[-1]
            return x[: , :,  : n // 2]
        def upsample (x):
            """
            Super simple upsampling
            """
            # Upsampling
            # Unsolved, Fabi has some basic ideas
            # Knn, interpolate?
            double_tensor = lambda x: tf.concat([x,x], axis=1)
            return tf.map_fn(double_tensor, x)

        
        with tf.name_scope("Encoder"):
            net = flex_convolution(net, pos, neighbors, 8, name="Conv8", activation=tf.nn.relu)
            net = flex_pooling(net, neighbors)
            net = subsample(net)
            pos = subsample(pos)
            neighbors = knn_bruteforce(pos, K=8)
            net = flex_convolution(net, pos, neighbors, 16, name="Conv16", activation=tf.nn.relu)
            net = flex_pooling(net, neighbors)
            net = subsample(net)
            pos = subsample(pos)
            neighbors = knn_bruteforce(pos, K=8)
            net = flex_convolution(net, pos, neighbors, 32, name="Conv32", activation=tf.nn.relu)

        with tf.name_scope ("Decoder"):
            net = flex_convolution_transpose(net, pos, neighbors, 16, name="Deconv16", activation=tf.nn.relu)
            net = flex_pooling(net, neighbors)
            net = upsample(net)
            pos = upsample(pos)
            neighbors = knn_bruteforce(pos, K=8)
            net = flex_convolution_transpose(net, pos, neighbors, 8, name="Deconv8", activation=tf.nn.relu)
            net = flex_pooling(net, neighbors)
            net = upsample(net)
            pos = upsample(pos)
            neighbors = knn_bruteforce(pos, K=8)

        net = flex_convolution_transpose(net, pos, neighbors, 3, name='reconstructed_points')
        total_cost = tf.reduce_mean(tf.square(tf.subtract(net,positions)),name="square_difference")
        summary.add_moving_summary(total_cost)
        return total_cost


    def optimizer(self):
        return tf.train.AdamOptimizer(1e-4)

class ChildModel(Model):
    def __init__(self, abc, **kwargs):
        self.abc = abc
        print self.abc
    def build_graph(self, positions):

        self.loss = self._build_graph(positions)

        return self.loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fusion', help='run sampling', default='') 
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.set_logger_dir('train_log/fusion_%s' % (args.fusion))

    #Loading Data
    df_train = get_point_cloud_dataflow('train',batch_size=BATCH_SIZE,num_points=PC["num"], model_ver="10",shuffle=False, normals=False)
    df_test = get_point_cloud_dataflow('test',batch_size=2*BATCH_SIZE,num_points=PC["num"], model_ver="10",shuffle=False, normals=False)
    steps_per_epoch = len(df_train)
    #Setup training step
    config = TrainConfig(
        model=ChildModel(),
        #data=FeedInput(df_train),
        dataflow = df_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(df_test, ScalarStats(['square_difference'])),
            ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=25,
        session_init=SaverRestore(args.load) if args.load else None
          )
    launch_train_with_config(config, SimpleTrainer())
