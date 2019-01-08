import os
import multiprocessing

from tensorpack.dataflow import (PrintData, BatchData, PrefetchDataZMQ, TestDataSpeed, MapData)
from tensorpack.utils import logger
from tensorpack.dataflow.serialize import LMDBSerializer

############################################
# Important! Set path to .lmdb data:
MODEL40PATH = '/graphics/scratch/students/heid/pointcloud_data/ModelNet40'
############################################

def get_modelnet_dataflow(
        name, batch_size=6,
        num_points=10000,
        parallel=None,
        model_ver="40",
        shuffle=False,
        normals=False,
        prefetch_data=True
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
    assert num_points in [1024, 10000]

    # Construct correct filename
    normals_str = ""
    if not normals:
        normals_str = "-positions"

    file_name = "model" + model_ver + "-" + name + normals_str + "-" + str(num_points) + ".lmdb"
    path = os.path.join(MODEL40PATH, file_name)

    # Try using multiple processing cores to load data
    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)
        logger.info("Using " + str(parallel) + " processing cores")

    # Construct dataflow object by loading lmdb file
    df = LMDBSerializer.load(path, shuffle=shuffle)

    #seperate df from labels and seperate into positions and vertex normals
    df = MapData(df, lambda dp: [ dp[1][:3], dp[1][3:] ] ) 

    if parallel < 16:
        logger.warn("DataFlow may become the bottleneck when too few processes are used.")
    if prefetch_data:
        df = PrefetchDataZMQ(df, parallel)

    if batch_size == 1:
        logger.warn("Batch size is 1. Data will not be batched.")
    df = BatchData(df, batch_size)
    df = PrintData(df)
    
    return df


if __name__ == '__main__':
    # Testing LMDB dataflow object
    # Load all different dataflow object types

    df = get_modelnet_dataflow('train', batch_size=2, num_points=1024, model_ver="40", normals=True,prefetch_data=False)


    for d in df:
        print " "
        print d[1].shape
        break
    # Test speed!
    #TestDataSpeed(df, 2000).start()
"""
    df = get_modelnet_dataflow('train', batch_size=8, num_points=10000, model_ver="10", normals=False)
    # Test speed!
    TestDataSpeed(df, 2000).start()

    df = get_modelnet_dataflow('train', batch_size=8, num_points=1024, model_ver="40", normals=False)
    # Test speed!
    TestDataSpeed(df, 2000).start()

    df = get_modelnet_dataflow('train', batch_size=8, num_points=10000, model_ver="40", normals=False)
    # Test speed!
    TestDataSpeed(df, 2000).start()

    df = get_modelnet_dataflow('train', batch_size=8, num_points=1024, model_ver="10", normals=True)
    # Test speed!
    TestDataSpeed(df, 2000).start()

    df = get_modelnet_dataflow('train', batch_size=8, num_points=10000, model_ver="10", normals=True)
    # Test speed!
    TestDataSpeed(df, 2000).start()
    df = get_modelnet_dataflow('train', batch_size=8, num_points=1024, model_ver="40", normals=True)
    # Test speed!
    TestDataSpeed(df, 2000).start()

    df = get_modelnet_dataflow('train', batch_size=8, num_points=10000, model_ver="40", normals=True)
    # Test speed!
    TestDataSpeed(df, 2000).start()
    """
