import numpy as np
import glob
import os

from tensorpack import *
import tensorflow as tf
from auto_encode import Model
from PointCloudDataFlow import PointCloudFromFile
from pc_meshlab_loader import *

def predict():
    prediction = PredictConfig(
            session_init = get_model_loader("train_log/fusion_/checkpoint"),
            model = Model(),
            input_names=['positions'],
            output_names=['reconstructed_points/Add']
            )
    pc_path = '/home/heid/Documents/master/pc2mesh/point_cloud_data/person_0001.txt'
    predictor = OfflinePredictor(prediction)
    pcff = PointCloudFromFile()
    pc_data = pcff.load_pc(path=pc_path,num_points=1024)
    outputs = predictor(pc_data)[0].T[:,:,0]
    print outputs.shape
    save_pc_to_file (outputs,'human.xyz','/home/heid/Documents/master/pc2mesh/point_cloud_data/')
    load_pc_meshlab('/home/heid/Documents/master/pc2mesh/point_cloud_data/human.xyz')
    

    

predict()
