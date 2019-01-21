#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ComputerGraphics Tuebingen, 2018


import tensorflow as tf
import numpy as np
import os
import time

# from __init__ import knn_bf_sym
#
from sklearn.neighbors import KDTree

#from __init__ import knn_bf_sym
from flex_conv_layers import knn_bf_sym

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

with tf.device('/device:GPU:0'):
    B = 2
    N = 4
    M = 4
    DP = 3
    K = 2
    # position = np.zeros((B, 2, N), dtype=np.float32)

    np.random.seed(0)
    # position = np.random.randint(0,9,[B,DP,N]).astype(np.float32)
    position_x = np.random.randn(B,DP,N).astype(np.float32) * 100
    print position_x
    position_y = np.random.randn(B,DP,M).astype(np.float32) * 100
    print position_y

    pos_x = tf.convert_to_tensor(position_x,tf.float32)
    pos_y = tf.convert_to_tensor(position_y,tf.float32)


    print "HERE#$$$$$$$$$$$$$$$$$$$$$$"
    actual_op = knn_bf_sym(pos_x,pos_y,K)
    print actual_op.shape.as_list()

    # Creates a session with log_device_placement set to True.
    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    sess = tf.Session()

    # print 'pos: ', position
    # print 'nbs: ', nbs

    ret = sess.run(actual_op)
    print np.shape(ret)
    ret = np.transpose(ret, [0, 2, 1])
    for i in ret:
        print i


    # ## timings
    # start = time.time()
    # for bi in range(B):
    #     pos = np.transpose(position[bi],[1,0])
    #     tree = KDTree(pos)
    #     dist,ind = tree.query(pos,k=K)
    # end = time.time()
    # print 'KD tree: time used in ms: ', (end - start)*1000.0
    #
    #

    for bi in range(B):
        posX = np.transpose(position_x[bi],[1,0])
        posY = np.transpose(position_y[bi],[1,0])
        print posX.shape
        # print pos
        #
        tree = KDTree(posY)
        dist,ind = tree.query(posX,k=2)

        print 'K@1 accuracy: ', np.sum(ind[:,1]
                == ret[bi][:,1])/float(N)
