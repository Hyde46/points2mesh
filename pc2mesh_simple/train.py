import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
from tensorpack import *
from tensorpack.input_source import QueueInput

from PointCloudDataFlow import get_modelnet_dataflow
from models import *
from fetcher import *


enable_argscope_for_module(tf.layers)

#TOTAL_BATCH_SIZE = 16
TOTAL_BATCH_SIZE = 1
BATCH_SIZE = 1
NUM_EPOCH = 25

PC = {'num': 1024, 'dp':3, 'ver':"40"}

seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer')
#flags.DEFINE_integer('feat_dim', 963, 'Number of units in perceptual featuer layer.')
flags.DEFINE_integer('feat_dim', 6, 'Number of units in FlexConv Feature layer')
flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rage.')
flags.DEFINE_integer('pc_num', 1024, 'Number of points per pointcloud object')
flags.DEFINE_integer('dp', 3, 'Dimension of points in pointcloud')
flags.DEFINE_integer('num_neighbors', 8, 'Number of neighbors considered during Graph projection layer')
flags.DEFINE_integer('batch_size', 1 , 'Batchsize')
flags.DEFINE_string('base_model_path', 'utils/ellipsoid/info_ellipsoid.dat', 'Path to base model for mesh deformation')

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
    df_train = get_modelnet_dataflow('train', batch_size=FLAGS.batch_size,
            num_points=PC["num"], model_ver=PC["ver"], shuffle=True, normals=True, prefetch_data=True)
    df_test = get_modelnet_dataflow('test', batch_size= 2 * FLAGS.batch_size,
            num_points=PC["num"], model_ver = PC["ver"], shuffle=True, normals=True, prefetch_data=True)
    steps_per_epoch = len(df_train)

    #Setup Model
    #Setup training step
    config = TrainConfig(
            model = pc2MeshModel(name="Pc2Mesh"),
            data = QueueInput(df_train),
            #data = FeedInput(df_train),
            callbacks=[
                ModelSaver(),
                #InferenceRunner(
                #    df_test)
                ],
            extra_callbacks=[
                MovingAverageSummary(),
                ProgressBar([]),
                MergeAllSummaries(),
                RunUpdateOps()
                ],
            steps_per_epoch=steps_per_epoch,
            max_epoch=NUM_EPOCH
            )
    #TODO:
    #Use SyncMultiGPUTrainer() when actually training the network
    launch_train_with_config(config, SimpleTrainer())
    #launch_train_with_config(config, SyncMultiGPUTrainer([0,1]))

