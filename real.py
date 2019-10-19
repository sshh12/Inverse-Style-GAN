import cv2
import time
import click
import pickle
import numpy as np
import tensorflow as tf

me = cv2.cvtColor(cv2.imread('real-shrivu.jpg'), cv2.COLOR_RGB2BGR)
me = np.rollaxis(me.reshape((1, 1024, 1024, 3)), 3, 1)

# From https://github.com/NVlabs/stylegan
import sys; sys.path.append('../stylegan')
from dnnlib import tflib; tflib.init_tf()
from dnnlib import util

import config
FACE_MODEL_URL = "https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ"


with util.open_url(FACE_MODEL_URL, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

class DummyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DummyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dummy_weights = self.add_weight("dummy_weights", shape=[1, 512], trainable=True, initializer=tf.initializers.random_normal)

    def call(self, input_):
        return self.dummy_weights

class GANLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs['trainable'] = False
        super(GANLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Gs = Gs.clone()

    def call(self, input_):
        return self.Gs.get_output_for(input_, None, is_validation=True, randomize_noise=True)

model = tf.keras.Sequential([
  DummyLayer(input_shape=(512,)),
  GANLayer()
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.MSE)

hist = model.fit(np.zeros((1, 512)), me, epochs=1000)

vec = model.layers[0].get_weights()[0]

np.save('test-real.npy', vec)
