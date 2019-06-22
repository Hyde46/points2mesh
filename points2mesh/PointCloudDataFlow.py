import os
import multiprocessing

from tensorpack.dataflow import (
    PrintData, BatchData, PrefetchDataZMQ, TestDataSpeed, MapData, JoinData)
from tensorpack.utils import logger
from tensorpack.dataflow.serialize import LMDBSerializer

import numpy as np

import random as rnd

############################################
# Important! Set path to .lmdb data:
MODEL40PATH = '/graphics/scratch/students/heid/pointcloud_data/ModelNet40'
############################################


def get_allowed_categories(version):
    """
    Modelnet40 categories
    0  -  airplane
    1  -  bathtub
    2  -  bed
    3  -  bench
    4  -  bookshelf
    5  -  bottle
    6  -  bowl
    7  -  car
    8  -  chair
    9  -  cone
    10 -  cup
    11 -  curtain
    12 -  desk
    13 -  door
    14 -  dresser
    15 -  flower_pot
    16 -  glass_box
    17 -  guitar
    18 -  keyboard
    19 -  lamp
    20 -  laptop
    21 -  mantel
    22 -  monitor
    23 -  night_stand
    24 -  person
    25 -  piano
    26 -  plant
    27 -  radio
    28 -  range_hood
    29 -  sink
    30 -  sofa
    31 -  stairs
    32 -  stool
    33 -  table
    34 -  tent
    35 -  toilet
    36 -  tv_stand
    37 -  vase
    38 -  wardrobe
    39 -  xbox
    """
    assert version in ["small", "big", "airplane", "chair", "sofa", "toilet"]
    if version == "big":
        return [0, 2, 5, 6, 7, 8, 17, 30, 35]
    if version == "small":
        return [0, 7, 8, 35]
    if version == "airplane":
        return [0]
    if version == "chair":
        return [8]
    if version == "sofa":
        return [20]
    if version == "toilet":
        return [35]
    return 0


def prepare_df(df, parallel, prefetch_data, batch_size):
    if parallel < 16:
        logger.warn(
            "DataFlow may become the bottleneck when too few processes are used.")
    if prefetch_data:
        df = PrefetchDataZMQ(df, parallel)
    if batch_size == 1:
        logger.warn("Batch size is 1. Data will not be batched.")
        return df
    df = BatchData(df, batch_size)
    return df


def get_modelnet_dataflow(
    name, batch_size=6,
    num_points=10000,
    parallel=None,
    model_ver="40",
    shuffle=False,
    normals=False,
    prefetch_data=False,
    noise_level=0.00
):
    """
    Loads Modelnet40 point cloud data and returns
    Dataflow tensorpack object consisting of a collection of points in 3d space given by (x,y,z) coordinates
    and normals for each point.
    Each lmdb datafile consists of:
    - [x, y, z] Data if normals=False
    - [x, y, z, nx, ny, nz] Data if normals=True
    The files consist of the following, depending on the filename:
    model{10,40}-{train,test}-{positions-}{1024,10000}.lmdb
    Modelnet generated the point cloud data by sampling many different objects 10000 times thus that many samples are
    available per object. This option of 10000 samples per object is set by default by :param num_points.
    To get faster loading and iteration times, a second option of 1024 is available.
    model10 - Consists of about ~4000 different models from 10 different categories
    model40 - Consists of about ~10000 different models from 40 different categories
    train and test data set are split 60/40
    if normals=True, the data set will include point normals and the resulting data will have the shape [6, num_points]
    otherwise only the 3d coordinates are included resulting in a data shape of [3, num_points]
    See Modelnet40 for more information on how the samples were generated, and object categories:
    [http://modelnet.cs.princeton.edu/]
    :param name: Determines wether to load train or test data. String in {'train','test'}.
     Throws exception if name is neither
    :param batch_size: Size of batches of Dataflow for SGD
    :param num_points: Number of samples per object. Either 1024 or 10000. Throws Exception if num_points is neither
    :param parallel: Number of cores used to prefetch data
    :param model_ver: Distinguishes between data set consisting of ~4000 objects or ~10000. String in {'10', '40'}
    :param shuffle: Wether to shuffle data or not for data flow
    :param normals: Determines if normals should be included in data or not. Boolean.
    :param prefetch_data: Determines whether to prefetch data with PrefetchDataZMQ or not
    :return: Dataflow object
    """
    # Check arguments
    assert batch_size > 0
    assert name in ['train', 'test']
    assert model_ver in ['10', '40']
    # Two different data sets exist with either 1024 samples per object or 10000 samples per object.
    # Different amounts of samples can still be used by choosing 10000 samples per object and selecting
    # a subset of them with the disadvantage of slower loading and sampling time.
    assert num_points in [1024, 7500, 10000]
    if num_points > 1024:
        file_num_points = 10000
    else:
        file_num_points = num_points
    # Construct correct filename
    normals_str = ""
    if not normals:
        normals_str = "-positions"

    file_name = "model" + model_ver + "-" + name + \
        normals_str + "-" + str(file_num_points) + ".lmdb"
    pass
    path = os.path.join(MODEL40PATH, file_name)

    # Try using multiple processing cores to load data
    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)
        logger.info("Using " + str(parallel) + " processing cores")

    allowed_categories = get_allowed_categories("big")

    # Construct dataflow object by loading lmdb file
    df = LMDBSerializer.load(path, shuffle=shuffle)

    # seperate df from labels and seperate into positions and vertex normals
    df = MapData(df, lambda dp: [[dp[1][:3] + (np.random.rand(3, file_num_points)*2*noise_level - noise_level)], [dp[1][3:]], [dp[1][:3]]]  # , dp[1][:3] + (np.random.rand(3,1024)*0.002 - 0.001)]
                 if dp[0] in allowed_categories else None)
    if num_points > 1024:
        df = MapData(df, lambda dp: np.array(
            dp)[:, :, :, 0:num_points].tolist())
    #df_noisy = noise_data_augmentation(df)
    df = prepare_df(df, parallel, prefetch_data, batch_size)
    df.reset_state()
    return df


if __name__ == '__main__':
    #get_advaced_mixed_modelnet_dataflow('train', '10k', batch_size=1)
    #df = LMDBSerializer.load("/graphics/scratch/datasets/ShapeNetCorev2/data/10/train_airplane_N10_S200.lmdb", shuffle=False)
    #for d in df:
    #    #print d[0]
    #    #print d[1]
    #   print np.concatenate((d[0], d[1]), axis=1)
    #   np.savetxt('/home/heid/tmp/test.asc',
    #           np.concatenate((d[0], d[1]), axis=1), delimiter=',')
    #   break
    df = get_modelnet_dataflow('train', batch_size=1, num_points=10000,
                               model_ver="40", normals=True, prefetch_data=False)
    for d in df:
        print np.array(d).shape
        print " "
        break
    # Test speed!
    #TestDataSpeed(df, 2000).start()

    """
    df = get_modelnet_dataflow(
        'train', batch_size=8, num_points=10000, model_ver="10", normals=False)
    # Test speed!
    TestDataSpeed(df, 2000).start()
    df = get_modelnet_dataflow(
        'train', batch_size=8, num_points=1024, model_ver="40", normals=False)
    # Test speed!
    TestDataSpeed(df, 2000).start()
        num_points=10000,
    df = get_modelnet_dataflow(
        'train', batch_size=8, num_points=10000, model_ver="40", normals=False)
    # Test speed!
    TestDataSpeed(df, 2000).start()
    df = get_modelnet_dataflow(
        'train', batch_size=8, num_points=1024, model_ver="10", normals=True)
    # Test speed!
    TestDataSpeed(df, 2000).start()
    df = get_modelnet_dataflow(
        'train', batch_size=8, num_points=10000, model_ver="10", normals=True)
    # Test speed!
    TestDataSpeed(df, 2000).start()
    df = get_modelnet_dataflow(
        'train', batch_size=8, num_points=1024, model_ver="40", normals=True)
    # Test speed!
    TestDataSpeed(df, 2000).start()
    df = get_modelnet_dataflow(
        'train', batch_size=8, num_points=10000, model_ver="40", normals=True)
    # Test speed!
    TestDataSpeed(df, 2000).start()
    """

    © 2019 GitHub, Inc.
    Terms
    Privacy
    Security
    Status
    Help

    Contact GitHub
    Pricing
    API
    Training
    Blog
    About

