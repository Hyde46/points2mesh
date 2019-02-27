import tensorflow as tf
from cd_dist import *
from user_ops import knn_bruteforce as _knn_bruteforce

def distance_density_loss(pred):
    _, distances, _ = _knn_bruteforce(tf.expand_dims(tf.transpose(pred),0), K=8)
    distances = distances[0]
    mean_neighborhood = tf.reduce_mean(distances,1)
    return tf.reduce_max(mean_neighborhood)
    

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
    return tf.nn.l2_normalize(tensor, dim=1)

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

    #edge length loss
    edge_length = tf.reduce_sum(tf.square(edge), 1)
    edge_loss = tf.reduce_mean(edge_length) * 300

    #chamfer distance
    dist1, idx1, dist2, idx2 = nn_distance(gt_pt, pred)
    point_loss = (chamfer_block_loss_metrics[block_id-1][0] * tf.reduce_mean(dist1)
            + chamfer_block_loss_metrics[block_id-1][1] * tf.reduce_mean(dist2)) * 3000

    #normal cosine loss
    normal = tf.gather(gt_nm, tf.squeeze(idx2, 0))
    normal = tf.gather(normal, placeholders['edges'][block_id - 1][:, 0])
    cosine = tf.abs(tf.reduce_sum(tf.multiply(unit(normal), unit(edge)), 1))
    normal_loss = tf.reduce_mean(cosine) * 0.5

    total_loss = point_loss + edge_loss +  normal_loss
    return total_loss