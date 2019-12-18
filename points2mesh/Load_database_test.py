import os
import multiprocessing

from tensorpack.dataflow import (
    PrintData, BatchData, PrefetchDataZMQ, TestDataSpeed, MapData)
from tensorpack.utils import logger
from tensorpack.dataflow.serialize import LMDBSerializer

import numpy as np

path = "/graphics/scratch/datasets/ModelNet40/advanced/1M/test_chair_N1000000_S200.lmdb"
df = LMDBSerializer.load(path, shuffle=False)

counter = 0
for d in df:
    counter += 1
    if counter % 15 == 0:
        print d[0].shape
        np.savetxt("/home/heid/tmp/"+str(counter) +
                   "test.xyz", d[0], delimiter=" ")
        print d[1].shape
