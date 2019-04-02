import numpy as np
import cPickle as pickle
import tensorflow as tf


def read_data(pkl):

    # Coord 3D coordinates of first stage ellipsoid
    # Shape (156, 3)
    coord = pkl[0]

    #
    # len 2
    # pool_idx[0] - (462, 2)
    # pool_idx[1] - (1848, 2)
    pool_idx = pkl[4]
    # len 3
    # faces[0] - (462, 4)
    # faces[1] - (1848, 4)
    # faces[2] - (7392, 4)
    faces = pkl[5]

    # len 3
    # lape_idx[0]
    # lape_idx[1]
    # lape_idx[2]
    lape_idx = pkl[7]
    print lape_idx[0][10]
    edges = []
    for i in range(1, 4):
        # len 3
        # pkl[0][1][0] - (1080, 2) - adjacent edges
        # pkl[1][1][0] - (4314, 2) - adjacent edges
        # pkl[2][1][0] - (17250, 2) - adjacent edges

        # Not Used:
        # pkl[0][1][1] - (1080,) - Not sure
        # pkl[1][1][1] - (4314,) - Not sure
        # pkl[2][1][1] - (17250,) - NOt sure

        # pkl[0][1][2] - (156,156) - tuple
        # pkl[1][1][2] - (618,618) - tuple
        # pkl[2][1][2] - (2466,2566) - tuple
        adj = pkl[i][1]
        edges.append(adj[0])


pkl = pickle.load(open(
    '/home/heid/Documents/master/pc2mesh/points2mesh/utils/ellipsoid/info_ellipsoid.dat', 'rb'))
fd = read_data(pkl)
