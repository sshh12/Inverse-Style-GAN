# Inverse Style GAN

> Looking up a generative latent vectors from reference images.

![example](https://user-images.githubusercontent.com/6625384/64915614-b82efd00-d730-11e9-92e4-f3a6de1a5575.png)

### Usage

```shell
$ git clone https://github.com/NVlabs/stylegan.git
$ git clone https://github.com/sshh12/Inverse-Style-GAN.git
$ cd Inverse-Style-GAN
// *Install Tensorflow* //
$ pip install -r requirements.txt
```

##### Brute Force Algo
```shell
$ python brute-lookup.py --img_path face.jpg --max_iter 100
```
This will iteratively output `best.jpg`/`best.npy` which correspond to the best matching generated image and its latent vector. 

### Related Research
* [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
* [Inverting The Generator Of A Generative Adversarial Network](https://arxiv.org/abs/1802.05701)
