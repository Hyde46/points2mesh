import tensorflow as tf
from inits import *
from flex_conv_layers import knn_bruteforce, knn_bf_sym
from cd_dist import *
import numpy as np
from timeit import default_timer as timer


a = np.random.rand(156, 3)
b = np.random.rand(1024, 3)

a = tf.convert_to_tensor(a, dtype=tf.float32)
b = tf.convert_to_tensor(b, dtype=tf.float32)

ak = np.random.rand(1, 3, 156)
bk = np.random.rand(1, 3, 1024)

ak = tf.convert_to_tensor(ak, dtype=tf.float32)
bk = tf.convert_to_tensor(bk, dtype=tf.float32)

dist1, idx1, dist2, idx2 = nn_distance(a, b)

knn, _, _ = knn_bf_sym(ak, bk, K=1)

sess = tf.Session()
start_nn = timer()
sess.run(dist1)
end_nn = timer()


start_knn = timer()
sess.run(knn)
end_knn = timer()

print "nn_distance speed: {0:.5f} seconds".format(end_nn - start_nn)
print "knn_bf_sym speed: {0:.5f} seconds".format(end_knn - start_knn)
