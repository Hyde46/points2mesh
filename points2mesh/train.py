import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
from tensorpack import *
from tensorpack.input_source import QueueInput
from tensorpack.dataflow import (PrintData, BatchData)

from PointCloudDataFlow import get_modelnet_dataflow
from models import *
from fetcher import *
from Idiss_df import *


enable_argscope_for_module(tf.layers)

# TOTAL_BATCH_SIZE = 16
TOTAL_BATCH_SIZE = 1
BATCH_SIZE = 1
NUM_EPOCH = 125

PC = {'num': 7500, 'dp': 3, 'ver': "40", 'gt': 10000}

seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
# basic Ellipsoid 156 - 618 - 2466
# basic torus     160 - 640 - 2560
# new ellipsoid   168 - 666 - 2658
# new ellipsoid   168 - 666 - 2658 - 10626
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer')
# flags.DEFINE_integer('feat_dim', 963, 'Number of units in perceptual featuer layer.')
flags.DEFINE_integer(
    'feat_dim', 483, 'Number of units in FlexConv Feature layer')

flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')
flags.DEFINE_float('collapse_epsilon', 0.008, 'Collapse loss epsilon')
# original 3e-5
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rage.')
flags.DEFINE_integer('pc_num', PC['num'], 'Number of points per pointcloud object')
flags.DEFINE_integer('dp', 3, 'Dimension of points in pointcloud')
flags.DEFINE_integer('feature_depth', 32,
                     'Dimension of first flexconv feature layer')
flags.DEFINE_integer(
    'num_neighbors', 6, 'Number of neighbors considered during Graph projection layer')
flags.DEFINE_integer('batch_size', 1, 'Batchsize')
#flags.DEFINE_string('base_model_path', 'utils/ellipsoid/info_ellipsoid.dat',
#                    'Path to base model for mesh deformation')
#flags.DEFINE_string('base_model_path', 'utils/ellipsoid/torus_small.dat',
#                    'Path to base model for mesh deformation')
flags.DEFINE_string('base_model_path', 'utils/ellipsoid/ellipsoid_2.dat',
                    'Path to base model for mesh deformation')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fusion', help='run sampling', default='')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    logger.set_logger_dir('/graphics/scratch/students/heid/train_log/true_c2_7500_big2_%s' % (args.fusion))

    # Loading Data
    df_train = get_modelnet_dataflow('train', batch_size=FLAGS.batch_size,
                                     num_points=PC["num"], model_ver=PC["ver"], shuffle=True, normals=True, prefetch_data=True,noise_level=0.0)
    df_test = get_modelnet_dataflow('test', batch_size=2 * FLAGS.batch_size,
                                    num_points=PC["num"], model_ver=PC["ver"], shuffle=True, normals=True, prefetch_data=True,noise_level=0.0)
    steps_per_epoch = len(df_train)

    # Setup Model
    # Setup training step
    config = TrainConfig(
        model=FlexmeshModel(PC, name="Flexmesh"),
        data=QueueInput(df_train),
        # data = FeedInput(df_train),
        callbacks=[
            ModelSaver(),
            MinSaver('total_loss'),
            # InferenceRunner(
            #    df_test,[])
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar([]),
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        steps_per_epoch=steps_per_epoch,
        starting_epoch=0,
        max_epoch=NUM_EPOCH
    )
    launch_train_with_config(config, SimpleTrainer())
    # TODO: Fix GPUTrainer
    # launch_train_with_config(config, SyncMultiGPUTrainerParameterServer([0],ps_device='gpu'))
