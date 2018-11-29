from tensorpack.dataflow.serialize import LMDBSerializer
from PointCloudDataFlow import *

#ds0 = PointCloudDataFlow('train',num_points=1024, load_ram=True, normals=False, model_ver="10")
#ds1 = PrefetchDataZMQ(ds0, nr_proc=2)
#LMDBSerializer.save(ds1, '/graphics/scratch/students/heid/pointcloud_data/ModelNet40/model10-train-positions-1024.lmdb')
#
#ds0 = PointCloudDataFlow('test',num_points=1024, load_ram=True, normals=False, model_ver="10")
#ds1 = PrefetchDataZMQ(ds0, nr_proc=2)
#LMDBSerializer.save(ds1, '/graphics/scratch/students/heid/pointcloud_data/ModelNet40/model10-test-positions-1024.lmdb')

ds0 = PointCloudDataFlow('train',num_points=1024, load_ram=True, normals=False, model_ver="40")
ds1 = PrefetchDataZMQ(ds0, nr_proc=2)
LMDBSerializer.save(ds1, '/graphics/scratch/students/heid/pointcloud_data/ModelNet40/model40-train-positions-1024.lmdb')

ds0 = PointCloudDataFlow('test',num_points=1024, load_ram=True, normals=False, model_ver="40")
ds1 = PrefetchDataZMQ(ds0, nr_proc=2)
LMDBSerializer.save(ds1, '/graphics/scratch/students/heid/pointcloud_data/ModelNet40/model40-test-positions-1024.lmdb')


ds0 = PointCloudDataFlow('train',num_points=10000, load_ram=True, normals=False, model_ver="10")
ds1 = PrefetchDataZMQ(ds0, nr_proc=2)
LMDBSerializer.save(ds1, '/graphics/scratch/students/heid/pointcloud_data/ModelNet40/model10-train-positions-10000.lmdb')

ds0 = PointCloudDataFlow('test',num_points=10000, load_ram=True, normals=False, model_ver="10")
ds1 = PrefetchDataZMQ(ds0, nr_proc=2)
LMDBSerializer.save(ds1, '/graphics/scratch/students/heid/pointcloud_data/ModelNet40/model10-test-positions-10000.lmdb')


ds0 = PointCloudDataFlow('train',num_points=10000, load_ram=True, normals=False, model_ver="40")
ds1 = PrefetchDataZMQ(ds0, nr_proc=2)
LMDBSerializer.save(ds1, '/graphics/scratch/students/heid/pointcloud_data/ModelNet40/model40-train-positions-10000.lmdb')

ds0 = PointCloudDataFlow('test',num_points=10000, load_ram=True, normals=False, model_ver="40")
ds1 = PrefetchDataZMQ(ds0, nr_proc=2)
LMDBSerializer.save(ds1, '/graphics/scratch/students/heid/pointcloud_data/ModelNet40/model40-test-positions-10000.lmdb')
