import numpy as np
import tensorflow as tf


def wrs_downsample_ids(survive_pobability, coarse_resolution):
    '''
    Args:
        survive_pobability: normalized probabilities [B,N]
        coarse_resolution: number of points to downsample
        max_attempts: possible rejections
    Return:
        ids: to downsample [B, coarse_resolution]
    '''
    B = tf.shape(survive_pobability)[0]
    N = int(survive_pobability.shape[1])

    u = tf.random_uniform([B, N])
    k = tf.pow(u, 1.0 / survive_pobability)

    sort = tf.contrib.framework.argsort(k, axis=-1, direction='DESCENDING')
    return tf.reshape(sort[:, :coarse_resolution], [B, coarse_resolution])


def downsample_by_id(x, ids):
    """Downsample point cloud features using specified ids obtained via
    'probability_downsample_ids'.
    Args:
        x (tf.tensor): features in fine resolution
        ids (tf.tensor): ids survive downsampling into coarse resolution

    Returns:
        tf.tensor: features in coarse resolution
    """
    xT = tf.transpose(x[:, :, :], [0, 2, 1])
    down = tf.batch_gather(xT, ids)
    return tf.transpose(down, [0, 2, 1])[:, :, :]
