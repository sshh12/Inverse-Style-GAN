"""
TODO
"""
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

from keras_vggface.vggface import VGGFace
from keras.models import Model as KerasModel
from keras.layers import Lambda as KerasLambda

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

class FaceFeatures(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs['trainable'] = False
        super(FaceFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        vggface_model = VGGFace(model='resnet50', include_top=True)
        feat_out = vggface_model.layers[-2].output
        self.model = KerasModel(vggface_model.input, feat_out)

    def call(self, input_):
        input_ = tf.transpose(input_, [0, 2, 3, 1])
        input_resized = tf.image.resize( 
            input_, 
            (224, 224), 
            method=tf.image.ResizeMethod.BICUBIC
        )
        return self.model(input_resized)

ref_model = tf.keras.Sequential([
  FaceFeatures()
])
ref_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MSE)
my_vec = ref_model.predict(me)

model = tf.keras.Sequential([
  DummyLayer(input_shape=(512,)),
  GANLayer(),
  FaceFeatures()
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.MSE)

for i in range(10000):
    hist = model.fit(np.empty((1, 512)), my_vec, epochs=500)
    loss = hist.history['loss'][-1]
    vec = model.layers[0].get_weights()[0]
    np.save('out\\test-real-{:.3f}.npy'.format(loss), vec)
