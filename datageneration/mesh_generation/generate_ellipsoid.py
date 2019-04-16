import numpy as np
import cPickle as pickle
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
#import tensorflow as tf

# Created Torus with
# (x*x + y*y + 0.5*z*z + 0.5*0.5 - 0.25*0.25)*(x*x + y*y + z*z + 0.5*0.5 - 0.25*0.25) - 4*0.5*0.5*(x*x + y*y)
# vertices (160, 3)
# faces (320, 3)


def normalize_adj(adj):
    print len(adj)
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    print adj.getnnz()
    print d_mat_inv_sqrt.getnnz()
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def chebyshev_polynomials(adj, k=3):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))

    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def process_vertex(line):
    ver = line.rstrip().split(" ")
    return ver


def process_face(line):
    f = line.rstrip().split(" ")
    assert int(f[0]) == 3
    return f[1:]


def read_torus(path):
    torus = []
    vertices = []
    faces = []
    with open(path, 'rU') as f:
        found_header = False
        vertices_end = False
        num_vertices = 0
        num_faces = 0
        processed_vertices = 0
        processed_faces = 0
        line_count = 0
        # Assume standard polygon file format
        for line in f:
            if line_count == 3:
                num_vertices = int(line[15:])
            if line_count == 7:
                num_faces = int(line[13:])
            if found_header and processed_vertices == num_vertices:
                faces.append(process_face(line))
                processed_faces += 1
            if found_header and processed_vertices < num_vertices:
                vertices.append(process_vertex(line))
                processed_vertices += 1
            found_header = line.rstrip() == 'end_header' or found_header
            line_count += 1
    assert len(vertices) == num_vertices and len(faces) == num_faces
    print "Succesfully read file"
    print "Calculating two step high resolution torus now"
    vertices = np.array(vertices)
    #print vertices
    faces = np.array(faces)
    torus = torus.append(vertices)


def read_data(pkl):

    # Coord 3D coordinates of first stage ellipsoid
    # Shape (156, 3)
    coord = pkl[0]
    # pool_idx[0] - (462, 2)
    # pool_idx[1] - (1848, 2)
    pool_idx = pkl[4]

    # len 3
    # lape_idx[0]
    # lape_idx[1]
    # lape_idx[2]
    lape_idx = pkl[7]
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
        #print adj[0]
        edges.append(adj[0])
    #print adj
    pol0 = chebyshev_polynomials(edges[0], 3)

pkl = pickle.load(open(
    '/home/heid/Documents/master/pc2mesh/points2mesh/utils/ellipsoid/info_ellipsoid.dat', 'rb'))
fd = read_data(pkl)
#read_torus("/home/heid/Documents/master/pc2mesh/datageneration/mesh_generation/torus.ply")
