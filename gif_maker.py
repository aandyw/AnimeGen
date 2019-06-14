import imageio
import os

path = 'DRAGAN/dcgan_dragan_images/'
images = []
for filename in os.listdir(path):
    if filename.startswith('epoch') and filename != 'epoch 0.png':
        filename = path + filename
        images.append(imageio.imread(filename))
imageio.mimsave('imgs/dcgan_dragan_result.gif', images)
