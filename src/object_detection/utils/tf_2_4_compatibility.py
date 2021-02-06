import tensorflow as tf
import sys

sys.setrecursionlimit(40000)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
