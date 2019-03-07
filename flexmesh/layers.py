from pixel2mesh.inits import *
import tensorflow as tf
from tensorpack.utils import logger
from flex_conv_layers import flex_convolution, flex_pooling, knn_bruteforce, knn_bf_sym

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=False,
                 sparse_inputs=False, act=tf.nn.relu, bias=True, gcn_block_id=1,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        if gcn_block_id == 1:
            self.support = placeholders['support1']
        elif gcn_block_id == 2:
            self.support = placeholders['support2']
        elif gcn_block_id == 3:
            self.support = placeholders['support3']

        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = 3  # placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphPooling(Layer):
    """Graph Pooling layer."""

    def __init__(self, placeholders, pool_id=1, **kwargs):
        super(GraphPooling, self).__init__(**kwargs)

        self.pool_idx = placeholders['pool_idx'][pool_id - 1]

    def _call(self, inputs):
        X = inputs

        add_feat = (1 / 2.0) * tf.reduce_sum(tf.gather(X, self.pool_idx), 1)
        outputs = tf.concat([X, add_feat], 0)

        return outputs

class GraphProjection(Layer):
    """Graph Projection layer."""

    def __init__(self, placeholders, **kwargs):
        super(GraphProjection, self).__init__(**kwargs)
        self.K = FLAGS.num_neighbors
        self.pc_feat = placeholders['pc_feature']

    def _call(self, inputs):
        stage_0 = self.mean_neighborhood(inputs,0) 
        stage_1 = self.mean_neighborhood(inputs,1)
        stage_2 = self.mean_neighborhood(inputs,2)
        stage_3 = self.mean_neighborhood(inputs,3)

        outputs = tf.concat([inputs, stage_0, stage_1, stage_2, stage_3], 1)
        return outputs

    def mean_neighborhood(self, inputs, num_feature):

        coord = inputs
        B = FLAGS.batch_size
        Dp = coord.shape.as_list()[1]
        #N = tf.constant(1,dtype=tf.int32)
        #N = coord.shape.as_list()[2]

        coord_expanded = tf.expand_dims( coord, -1)

        #transform PC feature to usable format
        pc_coords = self.pc_feat[num_feature]

        ellipsoid = tf.transpose(coord_expanded,[2, 1, 0])

        ellipsoid_N = ellipsoid.shape.as_list()[2]

        Y = tf.transpose(pc_coords, [0,2,1])

        N = ellipsoid.shape.as_list()[2]
        # Neighbors: [B, K, N]
        # Distances: [B, K, N]
        knn,_,_ = knn_bf_sym(ellipsoid, pc_coords, K=self.K)

        knnr = tf.reshape(knn, [1, B * N * self.K])

        #bv = tf.ones([B, N * self.K], dtype = tf.int32) * tf.expand_dims(tf.range(0,B), -1)
        bv = tf.ones([B,N*self.K],dtype=tf.int32) * tf.constant(np.arange(0,B),shape=[B,1],dtype=tf.int32)
        
        knnr = tf.stack([tf.reshape(bv,[1,B*N*self.K]), tf.reshape(knn,[1,B*N*self.K])],-1)

        knnY = tf.reshape(tf.gather_nd(Y, knnr), [B, N, self.K, Dp])
        knnY_mean = tf.reduce_mean(knnY, axis = 2)

        #ellipsoid = ellipsoid[0]

        return knnY_mean[0]

