import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import cv2
from models import *
from fetcher import * 
from pc_meshlab_loader import *
import re

enable_argscope_for_module(tf.layers)

seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

PC = {'num': 1024, 'dp': 3, 'ver': "40"}
#settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('pc', 'utils/examples/airplane_0627.txt', 'Test pointcloud')
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rage.')
flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer')
flags.DEFINE_integer('feat_dim', 239, 'Number of units in FlexConv Feature layer')
flags.DEFINE_integer('feature_depth', 32, 'Dimension of first flexconv feature layer')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')
flags.DEFINE_float('collapse_epsilon', 0.008, 'Collapse loss epsilon')
flags.DEFINE_integer('pc_num', 1024, 'Number of points per pointcloud object')
flags.DEFINE_integer('dp', 3, 'Dimension of points in pointcloud')
flags.DEFINE_integer('num_neighbors', 6, 'Number of neighbors considered during Graph projection layer')
flags.DEFINE_integer('batch_size', 1, 'Batchsize')
flags.DEFINE_string('base_model_path', 'utils/ellipsoid/info_ellipsoid.dat', 'Path to base model for mesh deformation')

num_blocks = 3
num_supports = 2

def load_pc(pc_path, num_points):
    #Load pointcloud from file
    data = np.genfromtxt(pc_path, delimiter=',')
    # strip away labels ( vertex normal )
    data = data[:num_points,0:3].T
    # Add single Batch [B,dp,N]
    data = data[np.newaxis,:,:]
    return data

def create_inference_mesh(vertices, num, pc,  path_to_input, output, display_mesh=False):

    vert = np.hstack((np.full([vertices.shape[0], 1], 'v'), vertices))
    face = np.loadtxt('utils/ellipsoid/face'+str(num)+'.obj', dtype='|S32')
    mesh = np.vstack((vert, face))

    result_name = pc.replace(".txt", "_result_p"+str(num)+".obj")
    path_to_mesh = os.path.join(output, result_name)
    np.savetxt(path_to_mesh, mesh, fmt='%s', delimiter=' ')

    if display_mesh:
        load_pc_meshlab(path_to_mesh)

def predict(predictor, data, path):
    vertices_1 = predictor(data)[0]
    vertices_2 = predictor(data)[1]
    vertices_3 = predictor(data)[2]
    return [vertices_1, vertices_2, vertices_3]

def loadModel():
    prediction = PredictConfig(
            session_init = get_model_loader("train_log/fusionGCN_/checkpoint"),
            model = FlexmeshModel(PC,name="Flexmesh"),
            input_names = ['positions'],
            output_names = ['mesh_outputs/output1',
                            'mesh_outputs/output2',
                            'mesh_outputs/output3'] 
            )
    #predict mesh
    predictor = OfflinePredictor(prediction)
    return predictor
 
def loadTxtFiles(path):
    pattern = re.compile(".*\.txt")
    files = [f for f in listdir(path) if isfile(join(path, f)) and pattern.match(f)]
    return files



path = "/home/heid/Documents/master/pc2mesh/point_cloud_data/small/"
#pcs = ["airplane_0627.txt", "airplane_0628.txt", "airplane_0629.txt", "bathtub_0146.txt", "car_0140.txt", "car_0160.txt", "car_0198.txt", "desk_0214.txt", "guitar_0188.txt", "person1.txt", "piano_0316.txt", "toilet1.txt", "toilet2.txt"]
pcs = loadTxtFiles(path)
path_output = "/home/heid/Documents/master/pc2mesh/points2mesh/utils/examples/results/"

predictor = loadModel()

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

for pc in pcs:

    path_pc = os.path.join(path, pc)
    pc_inp = load_pc(path_pc, num_points=1024)
    vertices = predict(predictor, pc_inp,path_pc)
    create_inference_mesh(vertices[2], 2, pc, path_pc, path_output, display_mesh=False)


