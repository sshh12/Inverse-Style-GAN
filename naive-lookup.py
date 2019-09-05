import sys
sys.path.append('../stylegan')

import pickle
import numpy as np
import cv2
import time
import click
import face_recognition

from dnnlib import tflib
from dnnlib import util
import dnnlib
import config


tflib.init_tf()


@click.command()
@click.option('--img',
    default='face.jpg',
    help='Path to query image.',
    type=click.Path())
@click.option('--output',
    default='best',
    help='Output file prefix.',
    type=click.Path())
@click.option('--max_iter',
    default=500,
    help='Maximum number of iterations.',
    type=int)
def find_face(img_path):
    pass
    
me = face_recognition.load_image_file("real-misha.jpg")
me_encoding = face_recognition.face_encodings(me)[0]

me = face_recognition.load_image_file("real-misha.jpg")
me_encoding = face_recognition.face_encodings(me)[0]
me2 = face_recognition.load_image_file("real-shrivu2.jpg")
me2_encoding = face_recognition.face_encodings(me2)[0]

rnd = np.random.RandomState(12)

best_score = 9999
best_img = None
best_latents = rnd.randn(1, 512)

cnt_last = 0

print('BASELINE=', face_recognition.face_distance([me_encoding], me2_encoding))

url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
with util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = pickle.load(f)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    
    
    while True:
        latents = best_latents * 0.6 + rnd.randn(1, 512) * 0.4
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        
        print(latents.shape, images.shape)
        cv2.imwrite('temp.jpg', cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
        
        gan = face_recognition.load_image_file("temp.jpg")
        try:
            gan_encoding = face_recognition.face_encodings(gan)[0]
        except IndexError:
            continue

        res = face_recognition.face_distance([me_encoding], gan_encoding)[0]

        if res < best_score:
            best_score = res
            best_img = images[0]
            best_latents = latents
            np.save('best.npy', best_latents)
            cv2.imwrite('best.jpg', cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
            cnt_last = 0
        else:
            cnt_last += 1
        print(res, best_score, cnt_last)

        cv2.imshow('best', cv2.cvtColor(best_img, cv2.COLOR_RGB2BGR))
        # cv2.imshow('cur', cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
        if 0xff & cv2.waitKey(5) == ord('q'):
            break

    cv2.destroyAllWindows()
    
