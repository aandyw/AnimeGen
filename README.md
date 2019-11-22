# AnimeLabs
An attempt to generate industry-standard facial images for anime characters using Generative Adversarial Networks

## Features
- [x] [Self Attention Module](https://arxiv.org/abs/1805.08318) + hinge loss
- [x] Conditional Batch Normalization
- [x] [Big GAN](https://arxiv.org/abs/1809.11096) + residual blocks
- [x] Instance Noise
- [X] Tensorboard visualization

## Results
**SAGAN V2 Epoch 217**  

![epoch 217 sagan_v2](https://github.com/Pie31415/Anime_GAN/blob/master/imgs/Epoch%20217.png)

**SAGAN with INSTANCE NOISE**

![sagan inst noise sample](https://github.com/Pie31415/AnimeLabs/blob/master/imgs/sagan_inst_sample.png)

## Losses

| **SAGAN V1**         | **SAGAN V2** |
| ------------- |:-------------:| 
| ![sagan v1 losses](https://github.com/Pie31415/Anime_GAN/blob/master/imgs/sagan.png)     | ![sagan v2 losses](https://github.com/Pie31415/Anime_GAN/blob/master/imgs/sagan_v2.png)| 

**SAGAN with INSTANCE NOISE**

![sagan inst noise losses](https://github.com/Pie31415/AnimeLabs/blob/master/imgs/sagan_inst.png)

## Prerequisites
- Python 3.7.3
- Pytorch 1.2.0
- Numpy 1.17.2

## Todo
- [ ] BCE Loss
- [ ] Tagging Data
- [ ] Get Big GAN to work?
