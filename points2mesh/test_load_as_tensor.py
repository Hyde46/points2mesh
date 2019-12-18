import tensorflow as tf
import os
import numpy as np

from fetcher import *


def load_ellipsoid_as_tensor():
    pkl = pickle.load(open('utils/ellipsoid/info_ellipsoid.dat', 'rb'))
    coord = pkl[0]
    pool_idx = pkl[4]
    faces = pkl[5]
    lape_idx = pkl[7]
    edges = []
    for i in range(1, 4):
        adj = pkl[i][1]
        edges.append(adj[0])

    neighbors = [[], [], []]
    neighbors[0] = [[] for i in range(156)]
    neighbors[1] = [[] for i in range(618)]
    neighbors[2] = [[] for i in range(2466)]
    i = 0
    for e in edges:
        for p in e:
            if p[0] == p[1]:
                continue
            neighbors[i][p[0]].append(p[1])
        i = i + 1

    # print neighbors

    # print neighbors

    # print lape_idx[0][0]
    # Coord

    #coord_ts = tf.convert_to_tensor(coord, dtype = tf.float32)

    # Support 1 - 3

    # print pkl[1][1][2]
    # print tf.convert_to_tensor_or_sparse_tensor(pkl[1][1])
   # s_1 = tf.convert_to_tensor(pkl[1][1][0], dtype=tf.int64)
   # s_2 = tf.convert_to_tensor(pkl[1][1][1], dtype=tf.float32)
   # s_3 = tf.convert_to_tensor(pkl[1][1][2], dtype=tf.int64)

   # sparse_support = tf.SparseTensor(indices=s_1, values=s_2, dense_shape=s_3)
   # sparse_support2 = tf.SparseTensor(indices=pkl[1][1][0], values=pkl[1][1][1], dense_shape=pkl[1][1][2])

    #support_x = [tf.convert_to_tensor(s[0],dtype=tf.float32) for s in pkl[1]]

    # print support_x

    # faces
    #faces_ts = [tf.convert_to_tensor(f) for f in faces]
    # edges
    #edges_ts = [tf.convert_to_tensor(t) for t in edges]

    # lape idx
    #lape_ts = [tf.convert_to_tensor(l) for l in lape_idx]

    # pool idx
    #pool_ts = [tf.convert_to_tensor(p) for p in pool_idx]
    # print pool_ts

    #feat = tf.gather(tf.convert_to_tensor(coord), pool_idx[0])

    #sess = tf.Session()
    # print sess.run(feat)


load_ellipsoid_as_tensor()
