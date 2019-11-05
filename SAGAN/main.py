from parameters import *

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    configs = get_parameters()
    main(configs)
