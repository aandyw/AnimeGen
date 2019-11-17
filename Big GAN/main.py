import os
import numpy as np
import matplotlib.pyplot as plt

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from parameters import *
from BigGAN import BigGAN


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(configs):

    # create the anime faces dataset
    dataset = dset.ImageFolder(root=configs.image_path,
                               transform=transforms.Compose([
                                   transforms.Resize(configs.imsize),
                                   transforms.CenterCrop(configs.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))

    # preparing dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=configs.batch_size,
        shuffle=True, num_workers=configs.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device : ", device)

    if configs.visualize:
        real_batch = next(iter(dataloader))

        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[
            :64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()

    # create folders to hold model, losses, and sample images
    make_folder(configs.model_path)
    make_folder(configs.sample_path)

    data_iter = iter(dataloader)
    real_images, real_labels = next(data_iter)
    num_classes = len(torch.from_numpy(np.unique(real_labels)))

    # using Big GAN model
    model = BigGAN(dataloader, num_classes, configs)
    if configs.train:
        model.train()

    if configs.plot:
        model.plot()


if __name__ == "__main__":
    configs = get_parameters()
    main(configs)
