import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

from layers import *
from generator import Generator
from discriminator import Discriminator


class BigGAN():
    """Big GAN"""

    def __init__(self, dataloader, num_classes, configs):

        self.dataloader = dataloader
        self.num_classes = num_classes

        # model settings & hyperparams
        # self.total_steps = configs.total_steps
        self.epochs = configs.epochs
        self.d_iters = configs.d_iters
        self.g_iters = configs.g_iters
        self.batch_size = configs.batch_size
        self.imsize = configs.imsize
        self.nz = configs.nz
        self.ngf = configs.ngf
        self.ndf = configs.ndf
        self.g_lr = configs.g_lr
        self.d_lr = configs.d_lr
        self.beta1 = configs.beta1
        self.beta2 = configs.beta2

        # instance noise
        self.inst_noise_sigma = configs.inst_noise_sigma
        self.inst_noise_sigma_iters = configs.inst_noise_sigma_iters

        # model logging and saving
        self.log_step = configs.log_step
        self.save_epoch = configs.save_epoch
        self.model_path = configs.model_path
        self.sample_path = configs.sample_path

        # pretrained
        self.pretrained_model = configs.pretrained_model

        # building
        self.build_model()

        # archive of all losses
        self.ave_d_losses = []
        self.ave_d_losses_real = []
        self.ave_d_losses_fake = []
        self.ave_g_losses = []

        if self.pretrained_model:
            self.load_pretrained()

    def build_model(self):
        """Initiate Generator and Discriminator"""
        self.G = Generator(self.nz, self.ngf, self.num_classes).cuda()
        self.D = Discriminator(self.ndf, self.num_classes).cuda()

        self.g_optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        print("Generator Parameters: ", parameters(self.G))
        print(self.G)
        print("Discriminator Parameters: ", parameters(self.D))
        print(self.D)
        print("Number of classes: ", self.num_classes)

    def load_pretrained(self):
        """Loading pretrained model"""
        checkpoint = torch.load(
            os.path.join(self.model_path, "{}_sagan.pth".format(
                self.pretrained_model)))

        # load models
        self.G.load_state_dict(checkpoint["g_state_dict"])
        self.D.load_state_dict(checkpoint["d_state_dict"])

        # load optimizers
        self.g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        self.d_optimizer.load_state_dict(checkpoint["d_optimizer"])

        # load losses
        self.ave_d_losses = checkpoint["ave_d_losses"]
        self.ave_d_losses_real = checkpoint["ave_d_losses_real"]
        self.ave_d_losses_fake = checkpoint["ave_d_losses_fake"]
        self.ave_g_losses = checkpoint["ave_g_losses"]

        print("Loading pretrained models (epoch: {})..!".format(
            self.pretrained_model))

    def reset_grad(self):
        """Reset gradients"""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self):
        """Train model"""
        step_per_epoch = len(self.dataloader)
        epochs = self.epochs
        total_steps = epochs * step_per_epoch

        # fixed z and labels for sampling generator images
        fixed_z = tensor2var(torch.randn(self.batch_size, self.nz))
        fixed_labels = tensor2var(
            torch.from_numpy(np.arange(self.num_classes)))

        print("Initiating Training")
        print("Epochs: {}, Total Steps: {}, Steps/Epoch: {}".
              format(epochs, total_steps, step_per_epoch))

        if self.pretrained_model:
            start_epoch = self.pretrained_model
        else:
            start_epoch = 0

        self.D.train()
        self.G.train()

        # Instance noise - make random noise mean (0) and std for injecting
        inst_noise_mean = torch.full(
            (self.batch_size, 3, self.imsize, self.imsize), 0).cuda()
        inst_noise_std = torch.full(
            (self.batch_size, 3, self.imsize, self.imsize), self.inst_noise_sigma).cuda()

        # total time
        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            # local losses
            d_losses = []
            d_losses_real = []
            d_losses_fake = []
            g_losses = []

            data_iter = iter(self.dataloader)
            for step in range(step_per_epoch):
                # Instance noise std is linearly annealed from self.inst_noise_sigma to 0 thru self.inst_noise_sigma_iters
                inst_noise_sigma_curr = 0 if step > self.inst_noise_sigma_iters else (
                    1 - step/self.inst_noise_sigma_iters)*self.inst_noise_sigma
                inst_noise_std.fill_(inst_noise_sigma_curr)

                # get real images
                real_images, real_labels = next(data_iter)
                real_images = real_images.cuda()

                # ================== TRAIN DISCRIMINATOR ================== #

                for _ in range(self.d_iters):
                    self.reset_grad()

                    # TRAIN REAL

                    # creating instance noise
                    inst_noise = torch.normal(
                        mean=inst_noise_mean, std=inst_noise_std).cuda()
                    # adding noise to real images
                    d_real = self.D(real_images + inst_noise, real_labels)
                    d_loss_real = loss_hinge_dis_real(d_real)
                    d_loss_real.backward()

                    # TRAIN FAKE

                    # create fake images using latent vector
                    z = tensor2var(torch.randn(real_images.size(0), self.nz))
                    fake_images = self.G(z, real_labels)

                    # creating instance noise
                    inst_noise = torch.normal(
                        mean=inst_noise_mean, std=inst_noise_std).cuda()
                    # adding noise to fake images
                    # detach fake_images tensor from graph
                    d_fake = self.D(fake_images + inst_noise, real_labels)
                    d_loss_fake = loss_hinge_dis_fake(d_fake)
                    d_loss_fake.backward()

                    d_loss = d_loss_real + d_loss_fake

                # optimize D
                self.d_optimizer.step()

                # ================== TRAIN GENERATOR ================== #

                for _ in range(self.g_iters):
                    self.reset_grad()

                    # create new latent vector
                    z = tensor2var(torch.randn(real_images.size(0), self.nz))

                    # generate fake images
                    inst_noise = torch.normal(
                        mean=inst_noise_mean, std=inst_noise_std).cuda()
                    fake_images = self.G(z, real_labels)
                    g_fake = self.D(fake_images + inst_noise, real_labels)

                    # compute hinge loss for G
                    g_loss = loss_hinge_gen(g_fake)
                    g_loss.backward()

                self.g_optimizer.step()

                # logging step progression
                if (step+1) % self.log_step == 0:
                    # logging losses
                    d_losses.append(d_loss.item())
                    d_losses_real.append(d_loss_real.item())
                    d_losses_fake.append(d_loss_fake.item())
                    g_losses.append(g_loss.item())

                    # print out
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [{}], Epoch: [{}/{}], Step [{}/{}], g_loss: {:.4f}, d_loss: {:.4f},"
                          " d_loss_real: {:.4f}, d_loss_fake: {:.4f}".
                          format(elapsed, (epoch+1), epochs, (step + 1), step_per_epoch,
                                 g_loss, d_loss, d_loss_real, d_loss_fake))

            # logging average losses over epoch
            self.ave_d_losses.append(mean(d_losses))
            self.ave_d_losses_real.append(mean(d_losses_real))
            self.ave_d_losses_fake.append(mean(d_losses_fake))
            self.ave_g_losses.append(mean(g_losses))

            # epoch update
            print("Elapsed [{}], Epoch: [{}/{}], ave_g_loss: {:.4f}, ave_d_loss: {:.4f},"
                  " ave_d_loss_real: {:.4f}, ave_d_loss_fake: {:.4f},".
                  format(elapsed, epoch+1, epochs, self.ave_g_losses[epoch], self.ave_d_losses[epoch],
                         self.ave_d_losses_real[epoch], self.ave_d_losses_fake[epoch]))

            # sample images every epoch
            fake_images = self.G(fixed_z, fixed_labels)
            fake_images = denorm(fake_images.data)
            save_image(fake_images,
                       os.path.join(self.sample_path,
                                    "Epoch {}.png".format(epoch+1)))

            # save model
            if (epoch+1) % self.save_epoch == 0:
                torch.save({
                    "g_state_dict": self.G.state_dict(),
                    "d_state_dict": self.D.state_dict(),
                    "g_optimizer": self.g_optimizer.state_dict(),
                    "d_optimizer": self.d_optimizer.state_dict(),
                    "ave_d_losses": self.ave_d_losses,
                    "ave_d_losses_real": self.ave_d_losses_real,
                    "ave_d_losses_fake": self.ave_d_losses_fake,
                    "ave_g_losses": self.ave_g_losses
                }, os.path.join(self.model_path, "{}_sagan.pth".format(epoch+1)))

                print("Saving models (epoch {})..!".format(epoch+1))
