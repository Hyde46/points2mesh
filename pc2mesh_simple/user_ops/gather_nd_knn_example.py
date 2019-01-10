import tensorflow as tf
import numpy as np
from __init__ import knn_bf_sym
#from flex_conv_layers import knn_bf_sym

def p2p_loss2(XT,YT):

    # calc p2p loss from X to Y
    # X [B,N,3] prediction
    # Y [B,N,3] ground_truth
    
    sess = tf.Session()

    B,Dp,N = XT.get_shape().as_list()
    B = 1 
    K = 8

    '''
    print "B:"
    print B
    print "N:"
    print N
    '''

    
    X = tf.transpose(XT,[0,2,1])
    Y = tf.transpose(YT,[0,2,1])

    #XT = tf.expand_dims(XT[0],0)
    #YT = tf.expand_dims(YT[0],0)
    # Find k nearest neighbors and their distances to each other
    print "Knn bf sym input"
    print "XT shape:"
    print XT.shape.as_list()
    print "YT shape:"
    print YT.shape.as_list()
    knn,dist, _ = knn_bf_sym(XT,YT,K)
    # Reshape into more useful tensor in 1D for easier access    
    knnr = tf.reshape(knn,[1,B*N*K])
    #print "Knnr shape:"
    #print knnr.shape.as_list()
    bv = tf.ones([B,N*K],dtype=tf.int32) * tf.constant(np.arange(0,B),shape=[B,1],dtype=tf.int32).eval(session=sess)

    knnr = tf.stack([tf.reshape(bv,[1,B*N*K]), tf.reshape(knn,[1,B*N*K])],-1)
    #sess.run(knnr)
    
    knnY = tf.reshape(tf.gather_nd(Y,knnr),[B,N,K,Dp])
    knnY_mean = tf.reduce_mean(knnY, axis= 2)
        #knnX = tf.reshape(tf.gather_nd(X,knnr),[B,N,K,Dp])
    #result = tf.stack([knnY,knnX],-1)
    print "result shape:"
    print knnY.shape.as_list()
    print "Mean shape:"
    print knnY_mean.shape.as_list()
    #ret = sess.run(knnY_mean)
    #print ret
#    return
 #   dist_min = tf.linalg.norm(X - knnY[:,:,0,:],axis=-1)
 #   dist_max = tf.linalg.norm(X - knnY[:,:,2,:],axis=-1)
    '''     
    knn = tf.transpose(knn,[0,2,1])
    '''
    '''

    normal, sup = point_plane(YT,knn)
    normal = tf.nn.l2_normalize(normal, axis=[1])
    
    X_rel = XT - sup
    point_to_plane_dist = tf.abs(tf.reduce_sum(normal * X_rel, axis=1))
    
    point_to_plane_dist = tf.where(dist_max<0.1,point_to_plane_dist,dist_min)
    

    return tf.reduce_mean(tf.square(point_to_plane_dist))
    '''


YT = tf.constant(
        [
            [
                [1,2,3,6,8,4,2],[5,6,7,2,5,8,0],[9,10,11,2,8,4,8]
                ],
            [
                [13,14,15,6,9,0,5],[17,18,19,2,76,9,4],[21,22,23,26,87,0,0]
                ]
            ],dtype=tf.float32
        )
XT = tf.constant(
        [
            [
                [1.1,2.4,3.6,4.8],[5.1,6.4,7.6,8.8],[9.1,10.4,11.6,12.8]
                ],
            [
                [13,14,15,16],[17,18,19,20],[21,22,23,24]
                ]
            ],dtype=tf.float32
        )
XT = tf.placeholder(tf.float32, shape=(1, 3, 156))
YT = tf.placeholder(tf.float32, shape=(None, 3, 1024))
p2p_loss2(XT,YT)
