import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import cv2
from models import *
from fetcher import *
from pc_meshlab_loader import *
import re
import cPickle as pickle
from flex_conv_layers import knn_bf_sym

enable_argscope_for_module(tf.layers)

seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

PC = {'num': 1024, 'dp': 3, 'ver': "40"}
# settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'pc', 'utils/examples/airplane_0627.txt', 'Test pointcloud')
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rage.')
flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer')
flags.DEFINE_integer(
    'feat_dim', 239, 'Number of units in FlexConv Feature layer')
flags.DEFINE_integer('feature_depth', 32,
                     'Dimension of first flexconv feature layer')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')
flags.DEFINE_float('collapse_epsilon', 0.008, 'Collapse loss epsilon')
flags.DEFINE_integer('pc_num', 1024, 'Number of points per pointcloud object')
flags.DEFINE_integer('dp', 3, 'Dimension of points in pointcloud')
flags.DEFINE_integer(
    'num_neighbors', 6, 'Number of neighbors considered during Graph projection layer')
flags.DEFINE_integer('batch_size', 1, 'Batchsize')
flags.DEFINE_string('base_model_path', 'utils/ellipsoid/info_ellipsoid.dat',
                    'Path to base model for mesh deformation')
#flags.DEFINE_string('base_model_path', 'utils/ellipsoid/torus_small.dat',
#                    'Path to base model for mesh deformation')
#flags.DEFINE_string('base_model_path', 'utils/ellipsoid/ellipsoid.dat',
#                    'Path to base model for mesh deformation')

num_blocks = 3
num_supports = 2


def f_score(label, predict, dist_label, dist_pred, threshold):
    num_label = label.shape[0]
    num_predict = predict.shape[0]

    f_scores = []
    for i in range(len(threshold)):
        num = len(np.where(dist_label <= threshold[i])[0])
        recall = 100.0 * num / num_label
        num = len(np.where(dist_pred <= threshold[i])[0])
        precision = 100.0 * num / num_predict

        f_scores.append((2*precision*recall)/(precision+recall+1e-8))
    return np.array(f_scores)


def noise_augment(data, noise_level=0.00):
    #rnd = np.random.rand(3, 1024)*2*noise_level - noise_level
    #return data + rnd
    return data


def load_pc(pc_path, num_points):
    #Load pointcloud from file
    data = np.genfromtxt(pc_path, delimiter=',')
    # strip away labels ( vertex normal )
    data = data[:num_points, 0:3].T
    data = noise_augment(data)
    # Add single Batch [B,dp,N]
    data = data[np.newaxis, :, :]
    return data


def predict(predictor, data, path):
    #vertices_1 = predictor(data)[0]
    #vertices_2 = predictor(data)[1]
    #vertices_3 = predictor(data)[2]

    #return [vertices_1, vertices_2, vertices_3]
    return predictor(data)[2]


def loadModel():
    prediction = PredictConfig(
        session_init=get_model_loader(
            "train_log/fusionProjectionBig1024_/checkpoint"),
        model=FlexmeshModel(PC, name="Flexmesh"),
        input_names=['positions'],
        output_names=['mesh_outputs/output1',
                      'mesh_outputs/output2',
                      'mesh_outputs/output3']
    )
    #predict mesh
    predictor = OfflinePredictor(prediction)
    return predictor


def loadTxtFiles(path):
    pattern = re.compile(".*\.txt")
    files = [f for f in listdir(path) if isfile(
        join(path, f)) and pattern.match(f)]
    return files


#path = "/home/heid/Documents/master/pc2mesh/point_cloud_data/evaluation_set/big_class"
path = "/graphics/scratch/students/heid/pointcloud_data/ModelNet40/test_complete"
pcs = loadTxtFiles(path)
#path_output = "/home/heid/Documents/master/pc2mesh/points2mesh/utils/examples/results/big_class_1024_3/"
#path_output = "/graphics/scratch/students/heid/inference/big_class_1024_3"
log_output = "/graphics/scratch/students/heid/inference"
predictor = loadModel()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

test_size = len(pcs)
print "Evaluating " + str(test_size) + " objects"
classes = ["airplane", "chair", "guitar", "sofa",
           "toilet", "bed", "bottle", "bowl", "car"]
model_count = {i: 0 for i in classes}
sum_cd = {i: 0 for i in classes}
sum_f = {i: 0 for i in classes}
sum_emd = {i: 0 for i in classes}

min_f = {i: [10000000, 1000000000] for i in classes}
min_cd = {i: 10000000 for i in classes}
min_f_id = {i: [0, 0] for i in classes}
min_cd_id = {i: 0 for i in classes}

thresholds = [0.001, 0.002]

sess = tf.Session()
max_len = len(pcs)
counter = 0
for pc in pcs:
    obj_class, obj_number = pc.split('.')[0].split('_')

    path_pc = os.path.join(path, pc)
    pc_inp = load_pc(path_pc, num_points=1024)
    vertices = predict(predictor, pc_inp, path_pc)
    pred = vertices
    pred = np.transpose(np.array(pred))
    label = pc_inp
    pred = pred[np.newaxis, :, :]
    label_tensor = tf.convert_to_tensor(label, dtype=tf.float32)
    pred_tensor = tf.convert_to_tensor(pred, dtype=tf.float32)

    model_count[obj_class] += 1.0

    #Evaluate min dist from label to pred and pred to label
    _, op_dist_pred_label, _ = knn_bf_sym(pred_tensor, label_tensor, K=1)
    _, op_dist_label_pred, _ = knn_bf_sym(label_tensor, pred_tensor, K=1)

    dist_pred_label = np.squeeze(sess.run(op_dist_pred_label))
    dist_label_pred = np.squeeze(sess.run(op_dist_label_pred))

    cd = np.mean(dist_pred_label) + np.mean(dist_label_pred)

    f = f_score(np.transpose(np.squeeze(label)), np.transpose(
        np.squeeze(pred)), dist_pred_label, dist_label_pred, thresholds)

    if min_f[obj_class][0] > f[0]:
        min_f[obj_class][0] = f[0]
        min_f_id[obj_class][0] = obj_number
    elif min_f[obj_class][1] > f[1]:
        min_f[obj_class][1] = f[1]
        min_f_id[obj_class][1] = obj_number
    if min_cd[obj_class] > cd:
        min_cd[obj_class] = cd
        min_cd_id[obj_class] = obj_number
    sum_f[obj_class] += f
    sum_cd[obj_class] += cd
    counter += 1
    if counter == max_len / 2:
        print "halfway done"

log = open(os.path.join(log_output, "points2mesh_evaluation.txt"), 'a')
for i in classes:
    number = model_count[i]
    f = sum_f[i] / number
    cd = (sum_cd[i] / number) * 1000.0
    print i, int(number), f, cd
    print >> log, i, int(number), f, cd, min_f[i], min_cd[i], min_f_id[i], min_cd_id[i]

log.close()
sess.close()

print sum_f
print sum_cd

print min_f_id
print min_cd_id
