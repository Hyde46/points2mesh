import os
import argparse
import tensorflow as tf
import numpy as np
import cv2

from tensorpack import *
from flex_conv_layers import flex_convolution, flex_pooling, knn_bruteforce

enable_argscope_for_module(tf.layers)





class pc2MeshModel(ModelDesc):

    def __init__(self, placeholders, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwarg.get('name')

        logging = kwargs.get('logging', False)

        self.placeholders = {}
        self.layers = []

    def inputs(self):
    """
    Inputs for Model:
    - pointcloud [batch, dim_position, num_points]
    - gt_pointcloud
    """
        return [self.placeholders['positions'],
                self.placeholders['ground_truth']]

    def build_graph(self, positions, ground_truth):
        # Build Flex Conv Part
        self.build_flex_graph(positions, ground_truth) # PC features
        # Mesh Deformation with GCN
        self.layers.append(GraphProjection(placeholders=self.placeholders)) 
         



        # Add up losses
        total_cost = 0
        # Update Tensorboard visualisation
        
        return total_cost



    def optimizer(self):
        return tf.train.AdamOptimizer(1e-4)

    def build_flex_graph(self, positions, ground_truth):
        def subsample(x):
            """
            Simplistic data subsampling
            TODO: Replace with IDISS from paper
            """
            n = x.shape.as_list()[-1]
            return x[:, :, :n // 4]
        # First only try NN for features
        features = positions
        neighbors = knn_bruteforce(positions, K = 8 )

        x = features
        #Multi scale features
        msx = {}

        #Build layers
        #TODO: regularizer for each layer
        for stage in range(4):
            if stage > 0:
                x = flex_pooling(x, neighbors)
                x = subsample(x)
                positions = subsample(positions)
                neighbors = knn_bruteforce(positions, K = 8 )
            x = flex_convolution(x, positions, neighbors, 64 * 
                    (stage + 1), activation = tf.nn.relu)
            x = flex_convolution(x, positions, neighbors, 64 * 
                    (stage + 1), activation = tf.nn.relu)
            msx[stage] = x
        
        #TODO: check if its right
        self.placeholders.update({'pc_feature' : [tf.squeeze(msx[0]),tf.squeeze(msx[1]),tf.squeeze(msx[2]), tf.squeeze(msx[3])])
        #TODO:
        self.loss += 0






if __name__ == '__main__':
    print "Here"
