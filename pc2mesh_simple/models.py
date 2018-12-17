import os
import argparse
import tensorflow as tf
import numpy as np
import cv2

from tensorpack import *
from flex_conv_layers import flex_convolution, flex_pooling, knn_bruteforce
from layers import *

enable_argscope_for_module(tf.layers)

flags = tf.app.flags
FLAGS = flags.FLAGS


class pc2MeshModel(ModelDesc):

    def __init__(self, placeholders, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()

        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = placeholders

        self.layers = []
        self.activations = []

        self.output1 = None
        self.output2 = None
        self.output3 = None
        self.output_stage_1 = None
        self.output_stage_2 = None
        
        self.loss = 0
        

    def inputs(self):
        # May be as big of a list as i wish it to be
        # TODO all placeholders here
        return [self.placeholders['positions'],
                self.placeholders['labels']]

    def build_graph(self, positions, label):

        # Build graphs
        with tf.variable_scope("Pointcloud Features"):
            self.cost += build_flex_graph()

        build_gcn_graph()
        #build sequential resnet model

        #connect graph and get cost
        #TODO: Potentially find a different way to build the network
        eltwise = [3, 5, 7, 9, 11, 13, 19, 21, 23, 25, 27, 29, 35, 37, 39, 41, 43, 45]
        #shortcuts
        concat = [15, 31]
        self.activations.append(self.inputs)
        
        with tf.name_scope("Mesh Deformation"):
            # Iterate over GCN layers and connect them
            for idx, layer in enumerate(self.layers):
                hidden = layer (self.activations[-1])
                if idx in eltwise:
                    hidden = tf.add(hidden, self.activations[-2]) * 0.5
                if idx in concat:
                    hidden = tf.concat([hidden, self.activations[-2]], 1)
                self.activations.append(hidden)

        with tf.name_scope("Mesh outputs"):
            #define outputs for multi stage mesh views
            self.output1 = self.activations[15]
            unpool_layer = GraphPooling(placeholders=self.placeholders, pool_id=1)
            self.output_stage_1 = unpool_layer(self.output1)

            self.output2 = self.activations[31]
            unpool_layer = GraphPooling(placeholders=self.placeholders, pool_id = 2)
            self.output_stage_2 = unpool_layer(self.output2)

            self.output3 = activations[-1]

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
        self.vars = {var.name: var for var in variables}
        
        #return cost of graph
        self.cost += self.get_loss() 
        summary.add_moving_summary(self.cost)
        return self.cost

    def build_flex_graph(positions, pc_input):

        def subsample(x,factor=2):
            # Number of samples
            n = x.shape.as_list()[-1]
            return x[:, :, :n // factor]

        # Features for each point is its own position in space
        features = positions
        neighbors = knn_bruteforce(positions, K = 8 )

        # 3 x 1024
        x = features
        #Build layers
        x = flex_convolution(x, positions, neighbors, 16, activation = tf.nn.relu)
        xr = 0.001 * tf.nn.l2_loss(x)
        x = flex_convolution(x, positions, neighbors, 16, activation = tf.nn.relu)
        xr += 0.001 * tf.nn.l2_loss(x)
        # 16 x 1024
        x0=x
        x = flex_pooling(x, neighbors)
        x = subsample(x)
        positions = subsample(positiosn)
        neighbors = knn_bruteforce(positions, K=8)
        x = flex_convolution(x, positions, neighbors, 32, activation = tf.nn.relu)
        xr = 0.001 * tf.nn.l2_loss(x)
        x = flex_convolution(x, positions, neighbors, 32, activation = tf.nn.relu)
        xr = 0.001 * tf.nn.l2_loss(x)
        # 32 x 512
        x1=x
        x = flex_pooling(x, neighbors)
        x = subsample(x)
        positions = subsample(positiosn)
        neighbors = knn_bruteforce(positions, K=8)
        x = flex_convolution(x, positions, neighbors, 64, activation = tf.nn.relu)
        xr = 0.001 * tf.nn.l2_loss(x)
        x = flex_convolution(x, positions, neighbors, 64, activation = tf.nn.relu)
        xr = 0.001 * tf.nn.l2_loss(x)
        # 64 x 256
        x2 = x
        x = flex_pooling(x, neighbors)
        x = subsample(x)
        positions = subsample(positiosn)
        neighbors = knn_bruteforce(positions, K=8)
        x = flex_convolution(x, positions, neighbors, 128, activation = tf.nn.relu)
        xr = 0.001 * tf.nn.l2_loss(x)
        x = flex_convolution(x, positions, neighbors, 128, activation = tf.nn.relu)
        xr = 0.001 * tf.nn.l2_loss(x)
        # 128 x 128
        x3 = x

        #Get output stages
        #TODO:here
        self.placeholders.update({'pc_feature' : [tf.squeeze(x0),tf.squeeze(x1),tf.squeeze(x2), tf.squeeze(x3)])
        #self.placeholders.update({'pc_feature' : [tf.squeeze(features)]})
        #TODO
        #Loss L2 Maybe working, definitely check:
        loss = xr
        return loss

    def build_gcn_graph():
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        self.layers.append(GraphConvolution(input_dim=FLAGS.feat_dim,
                                            output_dim=FLAGS.hidden,
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
        # Mesh deformation block with G ResNet
        for _ in range(12):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,
                                                gcn_block_id=1,
                                                placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=FLAGS.coord_dim,
                                            act=lambda x: x,
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
        # second project block
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        self.layers.append(GraphPooling(placeholders=self.placeholders, pool_id=1))  # unpooling
        self.layers.append(GraphConvolution(input_dim=FLAGS.feat_dim + FLAGS.hidden,
                                            output_dim=FLAGS.hidden,
                                            gcn_block_id=2,
                                            placeholders=self.placeholders, logging=self.logging))
        # Mesh deformation block with G ResNet
        for _ in range(12):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,
                                                gcn_block_id=2,
                                                placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=FLAGS.coord_dim,
                                            act=lambda x: x,
                                            gcn_block_id=2,
                                            placeholders=self.placeholders, logging=self.logging))
        # third project block
        self.layers.append(GraphProjection(placeholders=self.placeholders))
        self.layers.append(GraphPooling(placeholders=self.placeholders, pool_id=2))  # unpooling
        self.layers.append(GraphConvolution(input_dim=FLAGS.feat_dim + FLAGS.hidden,
                                            output_dim=FLAGS.hidden,
                                            gcn_block_id=3,
                                            placeholders=self.placeholders, logging=self.logging))
        # Mesh deformation block with G ResNet
        for _ in range(13):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,
                                                gcn_block_id=3,
                                                placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=FLAGS.coord_dim,
                                            act=lambda x: x,
                                            gcn_block_id=3,
                                             placeholders=self.placeholders, logging=self.logging))

    def get_loss():

        loss =  mesh_loss(self.output1, self.placeholders, 1)
        loss += mesh_loss(self.output2, self.placeholders, 2)
        loss += mesh_loss(self.output3, self.placeholders, 3)
        loss += .3 * laplace_loss(self.inputs, self.output1, self.placeholders, 1)
        loss += laplace_loss(self.output_stage_1, self.output2, self.placeholders, 2)
        loss += laplace_loss(self.output_stage_2, self.output3, self.placeholders, 3)
        
        # GCN loss
        conv_layers = range(1, 15) + range(17, 31) + range(33, 48)
        for layer_id in conv_layers:
            for var in self.layers[layer_id].vars.vales():
                loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        return loss

    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

if __name__ == '__main__':
    print "Here"
