import tensorflow as tf
import numpy as np
from layers import (flex_convolution,
                    flex_pooling,
                    knn_bruteforce)
from tabulate import tabulate
import os
import gzip
import subprocess

def save_pc_to_file(data, file_name, path_to_file):
    '''
    Saves pointcloud data as .xyz file to disk
    data should be in a Nx3 format
    @param path_to_file: where to save the data to
    '''
    np.savetxt(os.path.join(path_to_file,file_name), data, delimiter = " ")

def load_pc_meshlab(path_to_file):
    prg = 'meshlab'
    cmd = [prg, path_to_file]

    return_code = subprocess.check_output(cmd)
    print return_code

def load_pc_data(path, kind='train'):
    """Load Pointcloud data of ShapeNet dataset"""
    data_path = os.path.join(path,
                                'shapenetcore_partanno_segmentation_benchmark_v0_normal.tar.gz')
    with gzip.open(data_path, 'rb') as dtpath:
        data = np.frombuffer(dtpath.read(), dtype=np.float32,
                offset=32)
    print data[0] 

dataset_path = '/graphics/scratch/students/heid/pointcloud_data/ShapeNetPart/ShapeNet/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
example_object = os.path.join(dataset_path,'02691156/2b1a867569f9f61a54eefcdc602d4520.txt')


#positions = np.random.randn(B, Dp, N).astype(np.float32)
positions_n = np.loadtxt(example_object, dtype=np.float32)[:,0:3]

B, Din, Dout, Dout2, Dp, N, K = 1, 2, 4, 8, 3, 1024, 5
N = np.shape(positions_n)[0]
#features = np.random.randn(B, Din, N).astype(np.float32)

positions = positions_n.T.reshape(Dp,N)[np.newaxis,...]
features = positions
features = tf.convert_to_tensor(features, name='features')
positions = tf.convert_to_tensor(positions, name='positions')

net = [features]

neighbors = knn_bruteforce(positions, K=5)

net.append(flex_convolution(net[-1],
    positions,
    neighbors,
    Dout,
    activation = tf.nn.relu))
net.append(flex_convolution(net[-1],
    positions,
    neighbors,
    Dout,
    activation = tf.nn.relu))

'''
net.append(flex_pooling(net[-1],neighbors))

features = net[-1][: ,: , :N // 2 ]
positions = features[:, :, :N // 2]
net.append(features)

neighbors = knn_bruteforce(positions, K=3)

net.append(flex_pooling(net[-1],neighbors))

features = net[-1][: ,: , :N // 2 ]
positions = features[:, :, :N // 2]
net.append(features)
'''
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
#    save_pc_to_file(outputs[-1].T,file_name,path)
#    load_pc_meshlab(os.path.join(path,file_name))


