import tensorflow as tf
from cd_dist import *
from user_ops import knn_bruteforce as _knn_bruteforce
from flex_conv_layers import knn_bf_sym

flags = tf.app.flags
FLAGS = flags.FLAGS


def tension_loss(pred, positions, placeholders, block_id):
    B = 1
    Dp = pred.shape.as_list()[1]
    N = positions.shape.as_list()[2]
    K = 6
    # Transform tensors to the form of [B, Dp, N]
    features = tf.expand_dims(tf.transpose(positions), axis=0)
    input_features = tf.expand_dims(tf.transpose(positions), axis=0)
    Y = tf.expand_dims(pred, axis=0)

    # Find Nearest neighbors
    knn, _, _ = knn_bf_sym(features, input_features, K=K)
    knnr = tf.reshape(knn, [1, B * N * K])

    bv = tf.ones([B, N*K], dtype=tf.int32) * tf.constant(np.arange(0, B),
                shape=[B, 1], dtype=tf.int32)

    knnr = tf.stack([tf.reshape(bv, [1, B*N*K]), tf.reshape(knn, [1, B*N*K])], -1)
    knnY = tf.reshape(tf.gather_nd(Y, knnr), [B, N, K, Dp])

    # Could filter out 2 neighboring vertices from which new vertex got
    # interpolated from, but they can be ignored as they cancel out each other
    knnY_mean = tf.reduce_mean(knnY, axis=2)[0]

    dir_knnY = tf.subtract(knnY_mean, positions)

    return 0

def collapse_loss(pred):
    # dist1, _, _, _  = nn_distance(pred,pred)
    p = tf.transpose(pred, [1, 0])
    p = tf.expand_dims(p, 0)
    _, dist, _, = knn_bf_sym(p, p, K=2)
    dist = tf.squeeze(dist)
    dist1 = tf.identity(dist)
    coll_loss = tf.map_fn(lambda x: tf.cond(
                                        tf.less(x[1], FLAGS.collapse_epsilon),
                                            lambda:1.0,
                                            lambda:0.0), dist1)
    return tf.reduce_sum(coll_loss)
   
def point2triangle_loss(pred, placeholders, block_id):
    #TODO
    # Test point2triangle loss somewhere somehow

    pass

def laplace_coord(pred, placeholders, block_id):
    vertex = tf.concat([pred, tf.zeros([1, 3])], 0)
    indices = placeholders['lape_idx'][block_id - 1][:, :8]
    weights = tf.cast(placeholders['lape_idx'][block_id - 1][:, -1], tf.float32)

    weights = tf.tile(tf.reshape(tf.reciprocal(weights), [-1, 1]), [1, 3])
    laplace = tf.reduce_sum(tf.gather(vertex, indices), 1)
    laplace = tf.subtract(pred, tf.multiply(laplace, weights))
    return laplace


def laplace_loss(pred1, pred2, placeholders, block_id):
    lap1 = laplace_coord(pred1, placeholders, block_id)
    lap2 = laplace_coord(pred2, placeholders, block_id)

    laplace_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(lap1, lap2)), 1)) * 1500

    move_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(pred1, pred2)), 1)) * 100
    move_loss = tf.cond(tf.equal(block_id, 1), lambda: 0., lambda: move_loss)
    return laplace_loss + move_loss

def unit(tensor):
    #return tf.nn.l2_normalize(tensor, dim=1)
    return tf.nn.l2_normalize(tensor, axis=1)

def mesh_loss(pred, positions, vertex_normals,placeholders, block_id):
    chamfer_block_loss_metrics = [
           [0.55, 1.0],[0.75, 0.6],[1.0, 0.55] 
            ]
    gt_pt = tf.transpose(positions[0],[1,0])
    gt_nm = tf.transpose(vertex_normals[0],[1,0])

    # edge in graph
    nod1 = tf.gather(pred, placeholders['edges'][block_id - 1][:, 0])
    nod2 = tf.gather(pred, placeholders['edges'][block_id - 1][:, 1])
    edge = tf.subtract(nod1, nod2)

    # edge length loss
    edge_length = tf.reduce_sum(tf.square(edge), 1)
    edge_loss = tf.reduce_mean(edge_length) * 300

    # chamfer distance
    dist1, idx1, dist2, idx2 = nn_distance(gt_pt, pred)
    point_loss = (chamfer_block_loss_metrics[block_id-1][0] * tf.reduce_mean(dist1)
            + chamfer_block_loss_metrics[block_id-1][1] * tf.reduce_mean(dist2)) * 3000

    # normal cosine loss
    normal = tf.gather(gt_nm, tf.squeeze(idx2, 0))
    normal = tf.gather(normal, placeholders['edges'][block_id - 1][:, 0])
    cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(normal), unit(edge)), 1))
    normal_loss = tf.reduce_mean(cosine) * 0.5

    total_loss = point_loss + edge_loss +  normal_loss
    return total_loss
