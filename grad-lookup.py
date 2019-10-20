"""
TODO
"""
import cv2
import time
import click
import pickle
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.autograph.set_verbosity(1)

me = cv2.cvtColor(cv2.imread('real-shrivu.jpg'), cv2.COLOR_RGB2BGR)
me = np.rollaxis(me.reshape((1, 1024, 1024, 3)), 3, 1)

me2 = cv2.cvtColor(cv2.imread('real-shrivu2.jpg'), cv2.COLOR_RGB2BGR)
me2 = np.rollaxis(me2.reshape((1, 1024, 1024, 3)), 3, 1)

# From https://github.com/NVlabs/stylegan
import sys; sys.path.append('../stylegan')
from dnnlib import tflib; tflib.init_tf()
from dnnlib import util

from keras_vggface.vggface import VGGFace
from keras.models import Model as KerasModel
from keras.layers import Lambda as KerasLambda, Flatten as KerasFlatten

import config
FACE_MODEL_URL = "https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ"

with util.open_url(FACE_MODEL_URL, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

class DummyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DummyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dummy_weights = self.add_weight("dummy_weights", shape=[1, 512], trainable=True, initializer=tf.initializers.random_normal(0, 1))

    def call(self, input_):
        return self.dummy_weights

class GANLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs['trainable'] = False
        super(GANLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Gs = Gs.clone()

    def call(self, input_):
        face_img = self.Gs.get_output_for(input_, None, is_validation=True, randomize_noise=False)
        scale = 255 / 2
        return face_img * scale + (0.5 + 1 * scale)

class FaceFeatures(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs['trainable'] = False
        super(FaceFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        vggface_model = VGGFace(model='vgg16', weights='vggface', include_top=True, input_shape=(224, 224, 3))
        feat_out = vggface_model.layers[-2].output
        # flat_out = KerasFlatten()(feat_out)
        self.model = KerasModel(vggface_model.input, feat_out)
        # self.model = vggface_model

    def call(self, input_):
        input_ = tf.transpose(input_, [0, 2, 3, 1])
        input_resized = tf.image.resize( 
            input_, 
            (224, 224), 
            method=tf.image.ResizeMethod.BICUBIC
        )
        input_adjusted = input_resized[..., ::-1]
        mean = tf.constant([93.5940, 104.7624, 129.1863], dtype=tf.float32)
        input_adjusted = input_adjusted - tf.reshape(mean, [1, 1, 1, 3])
        return self.model(input_adjusted)


ref_model = tf.keras.Sequential([
  FaceFeatures()
])
my_vec = ref_model.predict(me)
my_vec2 = ref_model.predict(me2)
print(my_vec)
print(my_vec.shape)
print('REF DIFF =', np.linalg.norm(my_vec - my_vec2))

model = tf.keras.Sequential([
  DummyLayer(input_shape=(512,)),
  GANLayer(),
  FaceFeatures()
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.MSE)

best = 9e9
for i in range(10000):
    hist = model.fit(np.empty((1, 512)), my_vec2, epochs=2000)
    last_loss = hist.history['loss'][-1]
    if last_loss < best:
        vec = model.layers[0].get_weights()[0]
        np.save('out\\test-real-{:.3f}.npy'.format(last_loss), vec)
        best = last_loss
