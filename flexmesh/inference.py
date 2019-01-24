import tensorflow as tf
import cv2
from models import *
from fetcher import * 
from pc_meshlab_loader import *

enable_argscope_for_module(tf.layers)

seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

#settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('pc', 'utils/examples/airplane_0627.txt', 'Test pointcloud')
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rage.')
flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer')
flags.DEFINE_integer('feat_dim', 15, 'Number of units in FlexConv Feature layer')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')
flags.DEFINE_integer('pc_num', 1024, 'Number of points per pointcloud object')
flags.DEFINE_integer('dp', 3, 'Dimension of points in pointcloud')
flags.DEFINE_integer('num_neighbors', 8, 'Number of neighbors considered during Graph projection layer')
flags.DEFINE_integer('batch_size', 1 , 'Batchsize')
flags.DEFINE_string('base_model_path', 'utils/ellipsoid/info_ellipsoid.dat', 'Path to base model for mesh deformation')

num_blocks = 3
num_supports = 2

model = pc2MeshModel(logging = True)

def load_pc(pc_path, num_points):
    #load pc here
    data = np.genfromtxt(pc_path, delimiter=',')
    # strip away labels ( vertex normal )
    data = data[:num_points,0:3].T
    data = data[np.newaxis,:,:]
    return data

def create_inference_mesh(vertices,num):
    vert = np.hstack((np.full([vertices.shape[0], 1], 'v'), vertices))
    face = np.loadtxt('utils/ellipsoid/face'+str(num)+'.obj', dtype='|S32')
    mesh = np.vstack((vert, face))

    path_to_mesh = path.replace(".txt", str(num)+".obj")
    np.savetxt(path_to_mesh, mesh, fmt='%s', delimiter=' ')

    #load_pc_meshlab(path_to_mesh)

def predict(data,path):
    prediction = PredictConfig(
            session_init = get_model_loader("train_log/fusion_/checkpoint"),
            model = pc2MeshModel(name="Pc2Mesh"),
            input_names = ['positions'],#vertex normals are for validation. only need positions
            output_names = ['mesh_deformation/graphconvolution_14/add',
                            'mesh_deformation/graphconvolution_28/add',
                            'mesh_deformation/graphconvolution_43/add'] #[output1,output2,output3]
            )

    #predict mesh
    predictor = OfflinePredictor(prediction)
    #vertices_1 = predictor(data)[0]
    #vertices_2 = predictor(data)[1]
    vertices_3 = predictor(data)[2]

    #create_inference_mesh(vertices_1,1)
    #create_inference_mesh(vertices_2,2)
    create_inference_mesh(vertices_3,3)



pcs = ["airplane_0627.txt", "airplane_0628.txt", "airplane_0629.txt", "bathtub_0146.txt", "car_0140.txt", "car_0160.txt", "car_0198.txt", "desk_0214.txt", "guitar_0188.txt", "person1.txt", "piano_0316.txt", "toilet1.txt", "toilet2.txt"]
#for pc in pcs:
pc = pcs[9]
path = "/home/heid/Documents/master/pc2mesh/flexmesh/utils/examples/"
path = path + pc
pc_inp = load_pc(path, num_points=1024)
predict(pc_inp,path)


