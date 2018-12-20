import tensorflow as tf
import cv2



seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('learning_rate', 3e-5, 'Initial learning rate.')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer')
flags.DEFINE_integer('feat_dim', 963, 'Number of units in perceptual featuer layer.')
flags.DEFINE_integer('hidden', 192, 'Number of units in hidden layer')
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')
