import tensorflow as tf
import numpy as np
from tensorpack import *
from tensorpack.input_source import QueueInput
from tensorpack.dataflow import (PrintData, BatchData)
from wrs_df import *
from tabulate import tabulate
from scipy.spatial.distance import pdist, squareform

np.random.seed(42)
tf.set_random_seed(42)


class FakePointCloud(object):
    """
    docstring for FakePointCloud
    """

    def __init__(self, B, N, K, Din, Dout, Dp, N2=1, scaling=1):
        super(FakePointCloud, self).__init__()
        assert K < N
        self.B = B
        self.N = N
        self.K = K
        self.Din = Din
        self.Dout = Dout
        self.Dp = Dp
        self.N2 = N2

        dtype = np.float64

        def find_neighbors(positions, K):
            # B, Dpos, N
            all_neighbors = []
            for batch in positions:
                distances = squareform(pdist(batch.T, 'euclidean'))
                all_neighbors.append(np.argsort(distances, axis=1)[:, :K])
            return np.array(all_neighbors).transpose(0, 2, 1)

        def random_values(shape):
            return np.random.randn(*shape).astype(np.float32)

        self.theta = random_values(
            [1, self.Dp, self.Din, self.Dout]).astype(dtype)
        self.bias = random_values([self.Din, self.Dout]).astype(dtype)

        self.position = random_values([self.B, self.Dp, self.N]).astype(dtype)
        self.features = random_values([self.B, self.Din, self.N]).astype(dtype)
        self.neighborhood = find_neighbors(
            self.position, self.K).astype(dtype=np.int32)

    def init_ops(self, dtype=np.float32):
        self.theta_op = tf.convert_to_tensor(self.theta.astype(dtype))
        self.bias_op = tf.convert_to_tensor(self.bias.astype(dtype))

        self.features_op = tf.convert_to_tensor(self.features.astype(dtype))
        self.position_op = tf.convert_to_tensor(self.position.astype(dtype))
        self.neighborhood_op = tf.convert_to_tensor(self.neighborhood)

    def expected_feature_shape(self):
        return [self.B, self.Din, self.N]

    def expected_output_shape(self):
        return [self.B, self.Dout, self.N]

def wrs_downsample_ids(self, survive_pobability, coarse_resolution):
        '''
        Args:
            survive_pobability: normalized probabilities [B,N]
            coarse_resolution: number of points to downsample
            max_attempts: possible rejections
        Return:
            ids: to downsample [B, coarse_resolution]
        '''
        B = tf.shape(survive_pobability)[0]
        N = int(survive_pobability.shape[1])

        u = tf.random_uniform([B, N])
        k = tf.pow(u, 1.0 / survive_pobability)

        sort = tf.contrib.framework.argsort(k, axis=-1, direction='DESCENDING')
        return tf.reshape(sort[:, :coarse_resolution], [B, coarse_resolution])

def downsample_by_id(self, x, ids):
    """Downsample point cloud features using specified ids obtained via
    'probability_downsample_ids'.
    Args:
        x (tf.tensor): features in fine resolution
        ids (tf.tensor): ids survive downsampling into coarse resolution

    Returns:
        tf.tensor: features in coarse resolution
    """
    xT = tf.transpose(x[:, :, 0], [0, 2, 1])
    down = tf.batch_gather(xT, ids)
    return tf.transpose(down, [0, 2, 1])[:, :, None]

def fake_pc_loader():
    for k in range(1):
        pc = FakePointCloud(B=1, N=6, K=3, Din=3, Dout=3, Dp=3)
        pc.init_ops(dtype=np.float32)
        yield np.array([pc.position, pc.features+10])


if __name__ == '__main__':
    # Generate point cloud
    df = DataFromGenerator(fake_pc_loader)

    df = WRSDataFlow(
        df, neighborhood_sizes=3, sample_sizes=[6, 3])
    df.reset_state()
    for d in df:
        # kdt_coarse = KDTree(d[0], leaf_size=16, metric='euclidean')
        # kdt_sparse = KDTree(d[4], leaf_size=16, metric='euclidean')
        # neighborhood = kdt_sparse.query(
            # kdt_coarse.data, k = 4, dualtree = False, return_distance = False)
        # print d[0]
        # print d[1]
        print d[2]
        print d[3]
        print " "
        # print d[4]
        # print d[5]
        print d[6]
        print d[7]
        print " "
        '''
       # print d[8]
        print d[9]
        print d[10]
        print d[11]

        print ""

        print ""
    '''
