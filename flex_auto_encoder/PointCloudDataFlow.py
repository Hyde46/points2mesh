import os
import cv2
import numpy as np
import tqdm
import multiprocessing
import tensorflow as tf
from abc import abstractmethod

from tensorpack.input_source import QueueInput, StagingInput
from tensorpack.dataflow import (
    dataset, PrefetchDataZMQ, PrintData,
    BatchData, MultiThreadMapData, DataFlow,
    DataFromGenerator, PrefetchDataZMQ, TestDataSpeed)
from tensorpack.dataflow.dataset import Mnist
from tensorpack.utils import logger
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow.serialize import LMDBSerializer

import sys
import re as re

MODEL40PATH = '/graphics/scratch/students/heid/pointcloud_data/ModelNet40/'

tti={}


def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben

class PointCloudFromFile():

    def load_pc(self, path, num_points=1024):
        assert num_points in [1024,10000]
        self.shape = [1,3,num_points]
        data = np.loadtxt(path, delimiter=',', dtype=np.float32 )[:,0:3].T
        return data[np.newaxis]

def object_type_to_id(object_string):
    return tti[object_string]

def prepare_type_to_id_dict(directory):
    path = os.path.join(directory,"modelnet40_shape_names.txt")
    with open(path, 'r') as f:
        counter = 0
        for line in f:
            line = line.rstrip()
            if line == '':
                continue
            tti[line]=counter
            counter = counter + 1
    return tti


class PointCloudDataFlow(RNGDataFlow):
    """
    produces [label_id x y z nx ny nz] as point cloud data
    """
    PC_PATH = '/graphics/scratch/students/heid/pointcloud_data/ModelNet40/modelnet40_normal_resampled/'
    MODEL40 = 'modelnet40_'
    MODEL10 = 'modelnet10_'
    MODEL0  = 'modelnet0_'
    MODEL   = 'modelnet'

    def __init__(self, train_or_test,num_points=1024, load_ram=True, shuffle=False, directory=None, normals=True,model_ver="40" ):
        assert train_or_test in ['train', 'val', 'test']
        if normals:
            self.shape = [1,6,num_points]
        else:
            self.shape = [1, 3, num_points]

        self.type_to_id = prepare_type_to_id_dict(self.PC_PATH)

        self.load_ram = load_ram
        file_name = self.MODEL+model_ver+"_" + train_or_test + '.txt'
        file_name = os.path.join(self.PC_PATH,file_name)
        self.file_names = []
        self.shuffle = shuffle
        self.num_points = num_points
        self.current_idx = 0
        self.finished = True
        f = open(file_name)

        num_lines = sum(1 for line in open(file_name))
        if load_ram:
            logger.info("Loading Dataset into RAM")

        self.ds = np.zeros(self.shape)
        self.type_indices = []
        progress(0,num_lines)
        counter = 0
        while True:
            line = f.readline()
            line = line.strip()
            if not line:
                break
            object_type = "_".join(line.split("_")[0:-1])
            o_id = object_type_to_id(object_type)
            folder = os.path.join(self.PC_PATH,object_type)
            file_name = os.path.join(folder,line)

            path = os.path.join(folder,file_name)
            path += ".txt"
            self.file_names.append(path)
            if load_ram:
                if not normals:
                    data = np.loadtxt(path, delimiter=',', dtype=np.float32, skiprows=(10000-self.num_points))[:,0:3]
                else:
                    data = np.loadtxt(path, delimiter=',', dtype=np.float32, skiprows=(10000-self.num_points))
                #data = np.genfromtxt(path, delimiter=',', dtype=np.float32,skip_header=(10000-self.num_points))
                self.ds = np.append(self.ds,data.T[np.newaxis,:,:],axis=0)

                self.type_indices.append(o_id)
                counter += 1
                progress(counter,num_lines)


        self.ds = self.ds[1:]
        f.close()

    def __iter__(self):
        if self.finished is False:
            logger.error("PointCloudDataFlow not finished iterating. If you need to use the same dataflow in two places, you can simply create two dataflow instances.")
        self.finished = False
        idxs = list(range(self.__len__()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for self.idx in idxs:
            if not self.load_ram:
                f = self.file_names[self.idx]
                ds = np.loadtxt(f,delimiter=',')[ : , :self.num_points]
            else:
                ds = self.ds[self.idx,:,:]
            ds = [self.type_indices[self.idx],ds]
            yield ds
        self.finished = True

    def __len__(self):
        return len(self.file_names) 

def get_point_cloud_dataflow(
        name,batch_size=6,
        num_points=1024,
        parallel=None,
        model_ver="10",
        shuffle=False,
        normals=False
        ):
    assert name in ['train', 'val', 'test']
    assert num_points in [1024, 10000]
    isTrain = name == 'train'
    normals_str =""
    if not normals:
        normals_str = "-positions"
    file_name = "model"+model_ver+"-"+name+normals_str+"-"+str(num_points)+".lmdb"
    path = os.path.join(MODEL40PATH,file_name) 
    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)
        logger.info("Using "+str(parallel)+ " processing cores")
    if isTrain:
        df = LMDBSerializer.load(path, shuffle=shuffle)
        if parallel < 16:
            logger.warn("DataFlow may become the bottleneck when too few processes are used.")
        #df = PrefetchDataZMQ(df, parallel)
        df = PrintData(df)
        df = BatchData(df, batch_size)
    else:
        df = LMDBSerializer.load(path, shuffle=shuffle)
        if parallel < 16:
            logger.warn("DataFlow may become the bottleneck when too few processes are used.")
        #df = PrefetchDataZMQ(df, parallel)
        df = PrintData(df)
        df = BatchData(df, batch_size, remainder = False)

    return df
        

if __name__ == '__main__':
    #prepare_type_to_id_dict("/graphics/scratch/students/heid/pointcloud_data/ModelNet40/modelnet40_normal_resampled/")
    #df = PointCloudDataFlow('test',model_ver="10")
    #df = BatchData(df, 64)
    #TestDataSpeed(df,2000).start()
    # Test LMDB
    df = get_point_cloud_dataflow('train', batch_size=8, num_points=1024,model_ver="10", normals=True)
    #print len(df)
    TestDataSpeed(df, 2000).start()
    #mnist_data = dataset.Mnist('train',shuffle=False)

    #df_mnist = BatchData(mnist_data, 8)
    #for data in df:
    #    print " "
    #    print data[1].shape
    #    print " "


