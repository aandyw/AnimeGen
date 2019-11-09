import os
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image

from layers import *
from generator import Generator
from discriminator import Discriminator


class SAGAN():
    def __init__(self, dataloader, configs):

        # Data Loader
        self.dataloader = dataloader

        # model settings & hyperparams
        self.total_steps = configs.total_steps
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

        # model logging and saving
        self.log_step = configs.log_step
        self.model_path = configs.model_path
        self.log_path = configs.log_path
        self.sample_path = configs.sample_path

        # pretrained
        self.pretrained_model = configs.pretrained_model

        self.build_model()

        # archieve of all losses
        self.ave_d_loss = []
        self.ave_dr_loss = []
        self.ave_df_loss = []
        self.ave_g_loss = []
        self.ave_g_gamma1 = []
        self.ave_g_gamma2 = []
        self.ave_d_gamma1 = []
        self.ave_d_gamma2 = []

        if self.pretrained_model:
            self.load_pretrained()

    def build_model(self):
        # initialize Generator and Discriminator
        self.G = Generator(self.batch_size, self.imsize,
                           self.nz, self.ngf).cuda()
        self.D = Discriminator(self.batch_size, self.imsize, self.ndf).cuda()

        # optimizers
        self.g_optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        print("Generator Parameters: ", parameters(self.G))
        print(self.G)
        print()
        print("Discriminator Parameters: ", parameters(self.D))
        print(self.D)

    def load_pretrained(self):
        """Loading pretrained model"""
        checkpoint = torch.load(
            os.path.join(self.model_path, "{}_sagan.pth".format(
                self.pretrained_model)))
        # load models
        self.G.load_state_dict(checkpoint["gen_state_dict"])
        self.D.load_state_dict(checkpoint["disc_state_dict"])

        # load optimizers
        self.g_optimizer.load_state_dict(checkpoint["gen_optimizer"])
        self.d_optimizer.load_state_dict(checkpoint["disc_optimizer"])

        # load losses
        self.ave_d_loss = checkpoint["ave_d_loss"]
        self.ave_dr_loss = checkpoint["ave_dr_loss"]
        self.ave_g_loss = checkpoint["ave_g_loss"]
        self.ave_g_gamma1 = checkpoint["ave_g_gamma1"]
        self.ave_g_gamma2 = checkpoint["ave_g_gamma2"]
        self.ave_d_ganma1 = checkpoint["ave_d_ganma1"]
        self.ave_g_gamma2 = checkpoint["ave_g_gamma2"]

        print("Loading pretrained models (epoch: {})..!".format(
            self.pretrained_model))

    def reset_grad(self):
        """Reset gradients"""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self):
        step_per_epoch = len(self.dataloader)
        epochs = int(self.total_steps / step_per_epoch)

        # fixed z for sampling generator images
        fixed_z = tensor2var(torch.randn(self.batch_size, self.nz))

        print("Initiating Training")
        print("Epochs: {}, Total Steps: {}, Steps/Epoch: {}".
              format(epochs, self.total_steps, step_per_epoch))

        if self.pretrained_model:
            start_epoch = self.pretrained_model + 1
        else:
            start_epoch = 0

        # total time
        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            # local losses
            d_loss = []
            dr_loss = []
            df_loss = []
            g_loss = []
            g_gamma1 = []
            g_gamma2 = []
            d_gamma1 = []
            d_gamma2 = []

            data_iter = iter(self.dataloader)
            for step in range(step_per_epoch):
                # train layers
                self.D.train()
                self.G.train()

                # get real images
                real_images, _ = next(data_iter)
                real_images = tensor2var(real_images)

                # ================== TRAIN DISCRIMINATOR ================== #

                for _ in range(self.d_iters):
                    self.reset_grad()

                    # TRAIN REAL & FAKE IMAGES
                    # get D output for real images
                    d_real = self.D(real_images)

                    # generate fake images and get D output for fake images
                    z = tensor2var(torch.randn(real_images.size(0), self.nz))
                    fake_images = self.G(z)
                    d_fake = self.D(fake_images)

                    # compute hinge loss of D
                    d_loss_real, d_loss_fake = loss_hinge_dis(d_real, d_fake)
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()

                # optimize D
                self.d_optimizer.step()

                # ================== TRAIN GENERATOR ================== #

                for _ in range(self.g_iters):
                    self.reset_grad()

                    # create new latent vector
                    z = tensor2var(torch.randn(real_images.size(0), self.nz))

                    # generate fake images
                    fake_images = self.G(z)
                    g_fake = self.D(fake_images)

                    # compute hinge loss for G
                    g_loss_fake = loss_hinge_gen(g_fake)
                    g_loss_fake.backward()

                self.g_optimizer.step()

            if (step+1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], Epoch: [{}/{}], G_step [{}/{}], D_step[{}/{}], d_loss_real: {:.4f}, "
                      " d_loss_fake: {:.4f}, d_loss: {:.4f}, g_loss_fake: {:.4f}, g_loss".
                      format(elapsed, epoch+1, epochs, (step + 1), self.total_steps, (step + 1),
                             self.total_steps, d_loss_real,
                             d_loss_fake, g_loss_fake))
                # print(self.G.attn1.gamma)
                # print(self.G.attn2.gamma)

            # sample images every epoch
            fake_images = self.G(fixed_z)
            fake_images = denorm(fake_images.data)
            save_image(fake_images,
                       os.path.join(self.sample_path,
                                    "Epoch {}.png".format(epoch+1)))

            # save model every epoch
            torch.save({
                "gen_state_dict": self.G.state_dict(),
                "disc_state_dict": self.D.state_dict(),
                "gen_optimizer": self.g_optimizer.state_dict(),
                "disc_optimizer": self.d_optimizer.state_dict(),
                "ave_d_loss": self.ave_d_loss,
                "ave_dr_loss": self.ave_dr_loss,
                "ave_df_loss": self.ave_df_loss,
                "ave_g_loss": self.ave_g_loss,
                "ave_g_gamma1": self.ave_g_gamma1,
                "ave_g_gamma2": self.ave_g_gamma2,
                "ave_d_ganma1": self.ave_d_ganma1,
                "ave_g_gamma2": self.ave_g_gamma2
            }, os.path.join(self.model_path, "{}_sagan.pth".format(epoch+1)))

            print("Saving models (epoch {})..!".format(epoch+1))
