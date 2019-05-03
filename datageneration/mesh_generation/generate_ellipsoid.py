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


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    #print adj
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    #print "d inv sqrt shape"
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   # print d_inv_sqrt.shape
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose()\
        .dot(d_mat_inv_sqrt).tocoo()


def create_neighbor_matrix(adj, shape):
    neighbors = np.zeros(shape, dtype=int)
    for n in adj:
        if n[0] == n[1]:
            continue
        neighbors[n[0]][n[1]] = 1
        neighbors[n[1]][n[0]] = 1
    return neighbors


def chebyshev_polynomials(adj, k=3):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""

    #print adj
    #adj = np.transpose(adj)
    adj_normalized = normalize_adj(adj)
    #print "normalized shape"
    #print adj_normalized.shape

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
    ver = [float(ver[0]), float(ver[1]), float(ver[2])]
    #er = float(ver)
    return ver


def process_face(line):
    f = line.rstrip().split(" ")
    assert int(f[0]) == 3
    f = f[1:]
    f = [int(f[0]), int(f[1]), int(f[2])]
    return f


def get_edges(faces):
    edges = []
    print "Calculating edges ..."
    # Calculate edges for first step
    for f in faces:
        if [f[0], f[0]] not in edges:
            edges.append([f[0], f[0]])
        if [f[1], f[1]] not in edges:
            edges.append([f[1], f[1]])
        if [f[1], f[0]] not in edges:
            edges.append([f[1], f[0]])
        if [f[0], f[1]] not in edges:
            edges.append([f[0], f[1]])
        if [f[0], f[2]] not in edges:
            edges.append([f[0], f[2]])
        if [f[2], f[0]] not in edges:
            edges.append([f[2], f[0]])
        if [f[1], f[2]] not in edges:
            edges.append([f[1], f[2]])
        if [f[2], f[1]] not in edges:
            edges.append([f[2], f[1]])
    return edges


def get_pool_idx(edges):
    num_edges = np.max(edges) + 1
    print num_edges
    insert_check = np.zeros([num_edges, num_edges])
    pool_idx = []
    print " "
    print "Calculating pool_idx ..."
    for i in range(0, num_edges):
        for e in edges:
            if e[0] == e[1]:
                continue
            if insert_check[e[0]][e[1]] == 0 and insert_check[e[1]][e[0]] == 0:
                pool_idx.append(e)
                insert_check[e[0]][e[1]] = 1
                insert_check[e[1]][e[0]] = 1

    return pool_idx


def process_detail_step(faces):
    lape_idx = []

    edges = get_edges(faces)

    pool_idx = get_pool_idx(edges)
    num_edges = np.max(edges) + 1
    print " "
    print "Calculating lape_idx ..."
    for i in range(0, num_edges):
        neighbors = []
        neighbor_counter = 0
        for e in pool_idx:
            if e[0] == i:
                neighbors.append(e[1])
                neighbor_counter += 1
            if e[1] == i:
                neighbors.append(e[0])
                neighbor_counter += 1
        to_pad = 8 - neighbor_counter
        [neighbors.append(-1) for _ in range(0, to_pad)]
        neighbors.append(i)
        neighbors.append(neighbor_counter)
        lape_idx.append(neighbors)
    print " "
    print "Calculating support ..."
    edges_n = create_neighbor_matrix(
        edges, [np.max(edges) + 1, 1 + np.max(edges)])
    edges_n = nx.adjacency_matrix(nx.from_numpy_matrix(edges_n))
    support = chebyshev_polynomials(edges_n, 1)
    # Unpool torus
    unpooled_edges = {}
    next_vert_id = np.max(faces) + 1
    new_faces = []
    # Create new vertices and thus 4 new triangles
    # Add them to faces list
    appended_verts = []
    for f in faces:
        next_vert_id = add_new_edge(f[0], f[1], unpooled_edges, next_vert_id)
        next_vert_id = add_new_edge(f[1], f[2], unpooled_edges, next_vert_id)
        next_vert_id = add_new_edge(f[0], f[2], unpooled_edges, next_vert_id)
        zero_one_id = unpooled_edges[(f[0], f[1])]
        zero_two_id = unpooled_edges[(f[0], f[2])]
        one_two_id = unpooled_edges[(f[1], f[2])]
        new_faces.append([f[0], zero_one_id, zero_two_id])
        new_faces.append([f[1], one_two_id, zero_one_id])
        new_faces.append([f[2], one_two_id, zero_two_id])
        new_faces.append([zero_one_id, one_two_id, zero_two_id])
        
    return new_faces, edges, pool_idx, lape_idx, support


def add_new_edge(f1, f2, edges, next_vert_id):
    if (f1, f2) not in edges.keys():
        edges[(f1, f2)] = next_vert_id
        edges[(f2, f1)] = next_vert_id
        next_vert_id += 1

    return next_vert_id


def save_face(face, f_name):
    with open(f_name, 'a') as the_file:
        for f in face:
            to_write = "f {} {} {}\n".format(f[0] + 1, f[1] + 1, f[2] + 1)
            the_file.write(to_write)


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
    vertices = np.array(vertices)
    faces = np.array(faces)
    print vertices.shape
    new_faces, edges, pool_idx, lape_idx, support = process_detail_step(faces)
    new_faces2, edges2, pool_idx2, lape_idx2, support2 = process_detail_step(
        new_faces)
    new_faces3, edges3, pool_idx3, lape_idx3, support3 = process_detail_step(
        new_faces2)

    coord = vertices
    pool_idx = [pool_idx, pool_idx2]
    lape_idx = [lape_idx, lape_idx2, lape_idx3]
    #edges = [edges, edges2, edges3]
    faces = [faces, new_faces, new_faces2]
    torus = [coord, support, support2, support3,
             pool_idx, faces, 0, lape_idx]
    torus_file = "/home/heid/tmp/ellipsoid_2.dat"
    pickle.dump(torus, open(torus_file, 'wb'))

    face_counter = 1
    for f in faces:
        save_face(f, "/home/heid/tmp/face_ellipsoid_"+str(face_counter)+".obj")
        face_counter = face_counter + 1
    #support_3 = 0


def read_data(pkl):

    # Coord 3D coordinates of first stage ellipsoid
    # Shape (156, 3)
    coord = pkl[0]
    # pool_idx[0] - (462, 2)
    # pool_idx[1] - (1848, 2)
    pool_idx = pkl[4]
    #print coord
    # len 3
    # lape_idx[0]
    # lape_idx[1]
    # lape_idx[2]
    lape_idx = pkl[7]
    #print lape_idx[0][155]
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
        # print adj[0]
        edges.append(adj[0])
    edges_n = create_neighbor_matrix(
        edges[1], [np.max(edges[1]) + 1, 1 + np.max(edges[1])])
    edges_n = nx.adjacency_matrix(nx.from_numpy_matrix(edges_n))
    #print edges_n
    pol0 = chebyshev_polynomials(edges_n, 3)


pkl = pickle.load(open(
    '/home/heid/Documents/master/pc2mesh/points2mesh/utils/ellipsoid/info_ellipsoid.dat', 'rb'))
#fd = read_data(pkl)
read_torus(
    "/home/heid/Documents/master/pc2mesh/datageneration/mesh_generation/ellipsoid.ply")
