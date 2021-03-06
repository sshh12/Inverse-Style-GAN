import cv2
import click
import pickle
import numpy as np

# Tensorflow (tensorflow==1.14.0)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.autograph.set_verbosity(1)

# Keras (Keras==2.3.1)
from keras_vggface.vggface import VGGFace
from keras.models import Model as KerasModel
from keras.layers import Lambda as KerasLambda, Flatten as KerasFlatten

# From https://github.com/NVlabs/stylegan
import sys; sys.path.append('../stylegan')
from dnnlib import tflib; tflib.init_tf()
from dnnlib import util
import config
FACE_MODEL_URL = "https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ"


class DynamicInputLayer(tf.keras.layers.Layer):
    """
    An input layer that can we updated w/training.
    Ignores actual model input.

    Input --ignored--> DynamicInputLayer -> Random Vector
    """
    def __init__(self, init_value, **kwargs):
        super(DynamicInputLayer, self).__init__(**kwargs)
        self.init_value = init_value

    def build(self, input_shape):
        if self.init_value is None:
            initer = tf.initializers.random_normal(0, 1)
        else:
            initer = tf.constant_initializer(self.init_value)
        self.input_as_weights = self.add_weight("input_as_weights",
            shape=(1, 512), trainable=True, initializer=initer)

    def call(self, input_):
        return self.input_as_weights


class StyleGANLayer(tf.keras.layers.Layer):
    """
    A layer that implements StyleGAN.

    Latent Vec -> StyleGANLayer -> Image of Face (1024x1024)
    """
    def __init__(self, **kwargs):
        kwargs['trainable'] = False
        super(StyleGANLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        with util.open_url(FACE_MODEL_URL, cache_dir=config.cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
        self.Gs = Gs.clone()

    def call(self, input_):
        face_img = self.Gs.get_output_for(input_, None, 
            is_validation=True, randomize_noise=False)
        return (255 / 2) * face_img + 128


class FaceFeaturesLayer(tf.keras.layers.Layer):
    """
    A layer that converts a face into facial features.

    Face Image -> FaceFeaturesLayer -> Face Features (~2k feature vector)
    """
    def __init__(self, **kwargs):
        kwargs['trainable'] = False
        super(FaceFeaturesLayer, self).__init__(**kwargs)
        self.feat_mean = tf.reshape(tf.constant(
            [93.5940, 104.7624, 129.1863], dtype=tf.float32), [1, 1, 1, 3])

    def build(self, input_shape):
        vggface_model = VGGFace(model='vgg16', weights='vggface', 
            include_top=True, input_shape=(224, 224, 3))
        feat_out = vggface_model.layers[-2].output
        self.model = KerasModel(vggface_model.input, feat_out)

    def call(self, face_img):
        face_img = tf.transpose(face_img, [0, 2, 3, 1])
        img_resized = tf.image.resize( 
            face_img, 
            (224, 224), 
            method=tf.image.ResizeMethod.BICUBIC
        )
        return self.model(img_resized[..., ::-1] - self.feat_mean)


@click.command()
@click.option('--img_path',
    default='face.jpg',
    help='Path to query image',
    type=click.Path())
@click.option('--output',
    default='best',
    help='Output file prefix',
    type=click.Path())
@click.option('--input_init_vec',
    default=None,
    help='Use this file to init input layer. Should be (1,512) and an .npy',
    type=click.Path())
@click.option('--max_iter',
    default=100000,
    help='Maximum number of iterations',
    type=int)
@click.option('--lr',
    default=0.01,
    help='Learning rate',
    type=float)
@click.option('--seed',
    default=2,
    help='Randomness seed',
    type=int)
@click.option('--keras_verbose',
    default=0,
    help='Show Keras training logs',
    type=int)
def grad_descent_find_face(img_path, output, input_init_vec, max_iter, lr, seed, keras_verbose):
    """
    Iteratively find the closest face that can be generated by StyleGAN.

    Note 1: img_path must point to a square image of a face.
    Note 2: By default, this script with save the best img and its latent vector
        to best.jpg and best.npy.
    Note 3: This code tends to print a lot of warnings but if it gets to training
        then it's working.
    """
    np.random.seed(seed)
    tf.set_random_seed(seed)

    ref_img = cv2.cvtColor(cv2.resize(cv2.imread(img_path), (1024, 1024)), cv2.COLOR_BGR2RGB)
    ref_img = np.rollaxis(ref_img.reshape((1, 1024, 1024, 3)), 3, 1)

    vec_to_image = make_generator()

    # Compute target features
    ref_model = tf.keras.Sequential([
        FaceFeaturesLayer()
    ])
    ref_vec = ref_model.predict(ref_img)

    if input_init_vec is None:
        # None -> Totally random starting vector
        dynamic_init_weights = None
    else:
        dynamic_init_weights = np.load(input_init_vec)

    # Actual model for finding latent vector
    model = tf.keras.Sequential([
        DynamicInputLayer(dynamic_init_weights, input_shape=(1, 1)), # real input is ignored.
        StyleGANLayer(),
        FaceFeaturesLayer()
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MSE)

    print('#s: 0 = perfect, 0.01 = could be the same person, inf = worst')

    lowest_loss = float('inf')
    fit_attempts = max(max_iter // 1000, 1)
    for i in range(fit_attempts):
        print('Best ({}/{}) = {}'.format(i, fit_attempts, round(lowest_loss, 5)))
        hist = model.fit(np.empty((1, 1, 1)), ref_vec, epochs=1000, verbose=keras_verbose)
        hist_loss = hist.history['loss'][-1]
        if hist_loss < lowest_loss:
            best_vec = model.layers[0].get_weights()[0] # extract input latent vec
            np.save(output + '.npy', best_vec)
            cv2.imwrite(output + '.jpg', cv2.cvtColor(vec_to_image(best_vec)[0], cv2.COLOR_RGB2BGR))
            lowest_loss = hist_loss


def make_generator():
    """Make a simple func to convert latent vecs -> images"""
    with util.open_url(FACE_MODEL_URL, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        def gen(latents):
            images = Gs.run(latents, None, truncation_psi=0.7, 
                randomize_noise=True, 
                output_transform=fmt)
            return images
        return gen


if __name__ == '__main__':
    grad_descent_find_face()
