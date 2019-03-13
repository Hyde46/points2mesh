import os
import argparse
import tensorflow as tf
import numpy as np
import cv2

from tensorpack import *
from flex_conv_layers import flex_convolution, flex_pooling, knn_bruteforce
from layers import *
from losses import *
from fetcher import *

'''
Pointnet++ imports
'''
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append('/home/heid/Documents/master/pc2mesh/points2mesh/utils/pointnet_utils')
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module



enable_argscope_for_module(tf.layers)

flags = tf.app.flags
FLAGS = flags.FLAGS


class FlexmeshModel(ModelDesc):
    def __init__(self,PC, **kwargs):
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
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.output1 = None
        self.output2 = None
        self.output3 = None
        self.output_stage_1 = None
        self.output_stage_2 = None
        
        self.loss = 0
        self.cost = 0

        self.num_supports = 2
        self.num_blocks = 3
        self.PC = PC

        self.is_training = PC['is_training']
        self.bn_decay = PC['bn_decay']


    def inputs(self):
        return [tf.placeholder(tf.float32, (1, self.PC['dp'], self.PC['num']), "positions"),
                tf.placeholder(tf.float32, (1, self.PC['dp'], self.PC['num']), "vertex_normals"),
                ]

    def build_graph(self, positions, vertex_normals):
        
        self.load_ellipsoid_as_tensor()
        self.input = self.placeholders["features"]

        # Build graphs
        with tf.variable_scope("pointcloud_features"):
            self.cost += self.build_flex_graph(positions)
            #self.cost += self.build_pointnet(positions)

        self.build_gcn_graph(positions)

        #connect graph and get cost
        eltwise = [3, 5, 7, 9, 11, 13, 19, 21, 23, 25, 27, 29, 35, 37, 39, 41, 43, 45]
        eltwise = [e + 1 for e in eltwise]
        #shortcuts
        #concat = [15, 31]
        concat = [16, 32]
        self.activations.append(self.input)
        
        with tf.name_scope("mesh_deformation"):
            # Iterate over GCN layers and connect them
            for idx, layer in enumerate(self.layers):
                hidden = layer (self.activations[-1])
                if idx in eltwise:

                    hidden = tf.add(hidden, self.activations[-2]) * 0.5
                if idx in concat:
                    hidden = tf.concat([hidden, self.activations[-2]], 1)
                self.activations.append(hidden)

        with tf.name_scope("mesh_outputs"):
            #define outputs for multi stage mesh views
            #self.output1 = tf.identity(self.activations[15],name="output1")
            self.output1 = tf.identity(self.activations[16],name="output1")
            unpool_layer = GraphPooling(placeholders=self.placeholders,gt_pt=positions, pool_id=1)
            self.output_stage_1 = unpool_layer(self.output1)

            #self.output2 = tf.identity(self.activations[31],name="output2")
            self.output2 = tf.identity(self.activations[32],name="output2")
            unpool_layer = GraphPooling(placeholders=self.placeholders,gt_pt=positions, pool_id = 2)
            self.output_stage_2 = unpool_layer(self.output2)

            self.output3 = tf.identity(self.activations[-1],name="output3")

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
        self.vars = {var.name: var for var in variables}
        
        #return cost of graph
        self.cost += self.get_loss(positions,vertex_normals) 
        with tf.name_scope("loss_summaries"):
            tf.summary.scalar('total_loss', self.cost)

        return self.cost
   
    def build_pointnet(self, point_cloud):
        """ 
        Pointnet_part_seg model
        Input:
        pointcloud [B,N,Dp]
        Output:
        network, end_points
        """
        point_cloud = tf.transpose(point_cloud,[0,2,1])
        #batch_size = point_cloud.get_shape()[0].value
        batch_size = 1
        num_point = point_cloud.get_shape()[1].value
        end_points = {}
        l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
        l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,3])

        # Set Abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=self.is_training, bn_decay=self.bn_decay, scope='layer1')
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=self.is_training, bn_decay=self.bn_decay, scope='layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=self.is_training, bn_decay=self.bn_decay, scope='layer3')

        # Feature Propagation layers
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], self.is_training, self.bn_decay, scope='fa_layer1')
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], self.is_training, self.bn_decay, scope='fa_layer2')
        l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], self.is_training, self.bn_decay, scope='fa_layer3')

        # FC layers
        net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=self.is_training, scope='fc1', bn_decay=self.bn_decay)
        end_points['feats'] = net 
        net = tf_util.dropout(net, keep_prob=0.5, is_training=self.is_training, scope='dp1')
        net = tf_util.conv1d(net, 50, 1, padding='VALID', activation_fn=None, scope='fc2')

        return net, end_points

    def build_flex_graph(self, positions):

        def subsample(x,factor=4):
            # TODO: build better supsampling (IDISS)
            # Number of samples
            n = x.shape.as_list()[-1]
            return x[:, :, :n // factor]

        #xr = 0.001 * tf.nn.l2_loss(x)
        # Features for each point is its own position in space
        features = positions
        x = features
        neighbors,_,_ = knn_bruteforce(positions, K = 8 )
        x0 = features
        # Try not to use basic positions, but rather find important positions in pc
        #x0 = [positions,features]
        # feature 0 
        x = flex_convolution(x, positions, neighbors, FLAGS.feature_depth, activation = tf.nn.relu)
        x = flex_convolution(x, positions, neighbors, FLAGS.feature_depth, activation = tf.nn.relu)
        x = tf.identity(x, name="flex_layer_1")
        
        x1 = [positions, x]
        # Subsample!
        x = subsample(x)
        positions = subsample(positions)
        neighbors = knn_bruteforce(positions, K = 8)

        x = flex_convolution(x, positions, neighbors, FLAGS.feature_depth * 2, activation = tf.nn.relu)
        x = flex_convolution(x, positions, neighbors, FLAGS.feature_depth * 2, activation = tf.nn.relu)
        x = tf.identity(x, name="flex_layer_2")
        
        # Fully connected:
        x2 = [positions, x]
        #Subsample!
        x = subsample(x)
        positions = subsample(positions)
        neighbors = knn_bruteforce(positions, K = 8)

        x = flex_convolution(x, positions, neighbors, FLAGS.feature_depth * 4 , activation = tf.nn.relu)
        x = flex_convolution(x, positions, neighbors, FLAGS.feature_depth * 4, activation = tf.nn.relu)
        x = tf.identity(x, name="flex_layer_3")
        
        # Fully connected
        x3 = [positions, x]
        #Get output stages
        self.placeholders.update({'pc_feature' : [x0, x1, x2, x3]})

        return 0

    def build_gcn_graph(self,positions):
        self.layers.append(GraphAlignment(gt_pt=positions))
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
        self.layers.append(GraphPooling(placeholders=self.placeholders,gt_pt=positions, pool_id=1))  # unpooling for higher detail

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
        self.layers.append(GraphPooling(placeholders=self.placeholders,gt_pt=positions, pool_id=2))  # unpooling
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

    def get_loss(self,positions,vertex_normals):

        mesh_loss_first_block  = mesh_loss(self.output1, positions,vertex_normals,self.placeholders, 1)
        mesh_loss_second_block = mesh_loss(self.output2, positions, vertex_normals, self.placeholders, 2)
        mesh_loss_third_block  = mesh_loss(self.output3, positions, vertex_normals, self.placeholders, 3)
        
        #distance_loss0 = distance_density_loss(self.output1)
        #distance_loss1 = distance_density_loss(self.output2)
        #distance_loss2 = distance_density_loss(self.output3)

        with tf.name_scope("Mesh_loss"):
            summary.add_tensor_summary(mesh_loss_first_block ,['scalar'], name="mesh_loss")
            summary.add_tensor_summary(mesh_loss_second_block ,['scalar'], name="mesh_loss")
            summary.add_tensor_summary(mesh_loss_third_block ,['scalar'], name="mesh_loss")

        loss = mesh_loss_first_block + \
            mesh_loss_second_block +\
            mesh_loss_third_block

        l_loss_first = .3 * laplace_loss(self.input, self.output1, self.placeholders, 1)
        l_loss_second = laplace_loss(self.output_stage_1, self.output2, self.placeholders, 2)
        l_loss_third = laplace_loss(self.output_stage_2, self.output3, self.placeholders, 3)
        
        with tf.name_scope("laplacian_loss"):
            summary.add_tensor_summary(l_loss_first, ['scalar'], name="laplacian_loss")
            summary.add_tensor_summary(l_loss_second, ['scalar'], name="laplacian_loss")
            summary.add_tensor_summary(l_loss_third, ['scalar'], name="laplacian_loss")

        loss += l_loss_first + l_loss_second + l_loss_third
        '''
        
        with tf.name_scope("Collapse_loss"):
        #t_loss = tension_loss(self.output1, positions,self.placeholders, 1)
            c_loss = collapse_loss(self.output1)
            summary.add_tensor_summary(c_loss, ['scalar'],name ='collapse_loss')
            c_loss2 = collapse_loss(self.output2)
            summary.add_tensor_summary(c_loss, ['scalar'],name ='collapse_loss')
            c_loss3 = collapse_loss(self.output3)
            summary.add_tensor_summary(c_loss, ['scalar'],name ='collapse_loss')

        loss += c_loss + c_loss2 + c_loss3
        '''

        #with tf.name_scope("tension_loss"):
        #    pass
        #loss += distance_loss0 + distance_loss1 + distance_loss2
        loss = tf.identity(loss, name="complete_loss")


        
        # GCN loss
        conv_layers = range(1, 15) + range(17, 31) + range(33, 48)
        conv_layers = [e + 1 for e in conv_layers]
        for layer_id in conv_layers:
            for var in self.layers[layer_id].vars.values():
                loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        return loss

    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "utils/checkpoint/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

   
    def load_ellipsoid_as_tensor(self):
        pkl = pickle.load(open(FLAGS.base_model_path, 'rb'))
        coord = pkl[0]
        pool_idx = pkl[4]
        faces = pkl[5]
        lape_idx = pkl[7]
        edges = []
        #156 vertices
        for i in range(1, 4):
            adj = pkl[i][1]
            edges.append(adj[0])
        for i in range(0,3):
            lape_idx[i] = np.array(lape_idx[i])
            lape_idx[i][lape_idx[i] == -1] = np.max(lape_idx[i]) + 1

        # Define tensors based on loaded pkl object
        self.placeholders["features"] = tf.convert_to_tensor(coord, dtype = tf.float32)
        self.placeholders["support1"] = [ self.convert_support_to_tensor(s) for s in pkl[1]] 
        self.placeholders["support2"] = [ self.convert_support_to_tensor(s) for s in pkl[2]] 
        self.placeholders["support3"] = [ self.convert_support_to_tensor(s) for s in pkl[3]] 
        self.placeholders["faces"] = [tf.convert_to_tensor(f, dtype=tf.int32) for f in faces ]
        self.placeholders["edges"] = [tf.convert_to_tensor(e, dtype=tf.int32) for e in edges ]
        self.placeholders["lape_idx"] = [tf.convert_to_tensor(l, dtype = tf.int32) for l in lape_idx ]
        self.placeholders["pool_idx"] = [tf.convert_to_tensor(p, dtype = tf.int32) for p in pool_idx ]

        logger.info("Loaded Ellipsoid into Graph context")

    def convert_support_to_tensor(self, to_convert):
        indices = tf.convert_to_tensor(to_convert[0], dtype=tf.int64)
        values = tf.convert_to_tensor(to_convert[1], dtype=tf.float32)
        d_shape = tf.convert_to_tensor(to_convert[2], dtype=tf.int64)
        return tf.SparseTensor(indices=indices, values=values, dense_shape=d_shape)


if __name__ == '__main__':
    print "Dont run the model"
