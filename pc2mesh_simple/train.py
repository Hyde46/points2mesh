import tensorflow as tf
import cv2



seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('learning_rate', 3e-5, 'Initial learning rate.')
