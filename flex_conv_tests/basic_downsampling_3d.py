import tensorflow as tf
import numpy as np
from layers import (flex_convolution,
                    flex_pooling,
                    knn_bruteforce)
from tabulate import tabulate
import os
from pc_meshlab_loader import save_pc_to_file
from pc_meshlab_loader import load_pc_meshlab

B, Din, Dout, Dout2, Dp, N, K = 1, 2, 4, 8, 3, 1024, 5

features = np.random.randn(B, Din, N).astype(np.float32)
#positions = np.random.randn(B, Dp, N).astype(np.float32)
positions_n = np.loadtxt("/home/heid/Documents/master/point_cloud_data/mug.xyz", dtype=np.float32)
positions = positions_n.T.reshape(Dp,N)[np.newaxis,...]
features = positions
features = tf.convert_to_tensor(features, name='features')
positions = tf.convert_to_tensor(positions, name='positions')

net = [features]

neighbors = knn_bruteforce(positions, K=5)
'''
net.append(flex_convolution(net[-1],
    positions,
    neighbors,
    Dout,
    activation = tf.nn.relu))
'''
net.append(flex_pooling(net[-1],neighbors))
net.append(flex_pooling(net[-1],neighbors))
net.append(flex_pooling(net[-1],neighbors))
net.append(flex_pooling(net[-1],neighbors))
net.append(flex_pooling(net[-1],neighbors))
net.append(flex_pooling(net[-1],neighbors))

#features = net[-1][: ,: , :N // 2 ]
#positions = positions[:, :, :N // 2]
#features = positions
#net.append(features)


gradient_wrt_feature = tf.gradients(net[-1], net[0])


file_name = 'mug_sub.xyz'
path = '/home/heid/Documents/master/Flex-Convolution/out'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    outputs = sess.run(net[-1])
    grads = sess.run(gradient_wrt_feature)

    assert not np.isnan(outputs).any()
    assert not np.isnan(grads).any()

    print(tabulate([[v.name, v.shape] for v in tf.trainable_variables()],
      headers=["Name", "Shape"]))

    print(tabulate([[n.name, n.shape] for n in net], headers=["Name", "Shape"]))
    print outputs[-1]
    save_pc_to_file(outputs[-1].T,file_name,path)
    load_pc_meshlab(os.path.join(path,file_name))

