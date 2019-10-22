# Inverse Style GAN

> Looking up a generative latent vectors from reference images.

![example](https://user-images.githubusercontent.com/6625384/64915614-b82efd00-d730-11e9-92e4-f3a6de1a5575.png)

### Usage

```shell
$ git clone https://github.com/NVlabs/stylegan.git
$ git clone https://github.com/sshh12/Inverse-Style-GAN.git
$ cd Inverse-Style-GAN
$ echo Install Tensorflow For GPUs
$ pip install -r requirements.txt
```

#### Gradient Descent Algo
```shell
$ python grad-lookup.py --img_path face.jpg --max_iter 10000 --lr 0.1 --keras_verbose 1
```
or resume training given `best.npy` with
```shell
$ python grad-lookup.py --img_path face.jpg --max_iter 10000 --input_init_vec best.npy
```
This will iteratively output `best.jpg`/`best.npy` which correspond to the best matching generated image and its latent vector. 

##### How?

The script creates a model (and sort of a pipeline) that takes a latent vector, converts it to a face (StyleGan), and then finds the facial features of that face (FaceNet). Since GANs don't normally work backward, this script leverages the fact that both StyleGAN and VGGFace are differentiable to find the latent vector that would produce a given target face. When trained, the model freezes its StyleGAN and VGGFace weights so the only update on each iteration of gradient descent is the input latent vector. The model's loss is a function of the L2 difference between the target face's embeddings and the generated face's embeddings.

#### Brute Force Algo
```shell
$ python brute-lookup.py --img_path face.jpg --max_iter 100
```
This will also iteratively output `best.jpg`/`best.npy` which correspond to the best matching generated image and its latent vector. 

### Related Research
* [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
* [Inverting The Generator Of A Generative Adversarial Network](https://arxiv.org/abs/1802.05701)
* [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
