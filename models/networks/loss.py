"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import VGG19
import numpy as np


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.real_label_anchor = None
        self.fake_label_anchor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'categorical':
            pass
        elif gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_target_anchor(self, input, target_is_real):
        if target_is_real:
            unif = np.random.uniform(-1, 1, 1000)
            count, bins = np.histogram(unif, self.opt.num_outcomes)
            anchor1 = count / sum(count)
            self.real_label_anchor = self.Tensor(anchor1).unsqueeze(0)
            self.real_label_anchor.requires_grad_(False)
            return self.real_label_anchor.repeat(input.shape[0]*input.shape[2]*input.shape[3], 1)
        else:
            gauss = np.random.normal(0, 0.1, 1000)
            count, bins = np.histogram(gauss, self.opt.num_outcomes)
            anchor0 = count / sum(count)
            self.fake_label_anchor = self.Tensor(anchor0).unsqueeze(0)
            self.fake_label_anchor.requires_grad_(False)
            return self.fake_label_anchor.repeat(input.shape[0]*input.shape[2]*input.shape[3], 1)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True, feat_real=None):
        if self.gan_mode == 'categorical':
            if feat_real is None:
                target_anchor = self.get_target_anchor(input, target_is_real)
            else:
                print(feat_real)
                target_anchor = feat_real.transpose(1,-1).flatten(start_dim=0, end_dim=2).contiguous()
                target_anchor = target_anchor.log_softmax(1).exp()
            batch_size = input.shape[0]
            v_min = -1
            v_max = 1
            supports = torch.linspace(v_min, v_max, self.opt.num_outcomes).view(1, 1, self.opt.num_outcomes).cuda()
            delta = (v_max - v_min) / (self.opt.num_outcomes - 1)
            feat = input.transpose(1,-1).flatten(start_dim=0, end_dim=2).contiguous()
            feat = feat.log_softmax(1).exp()
            
            if target_is_real:
                skew = torch.zeros((batch_size, self.opt.num_outcomes)).cuda().fill_(1)
            else:
                skew = torch.zeros((batch_size, self.opt.num_outcomes)).cuda().fill_(-1)
            Tz = skew + supports.view(1, -1) * torch.ones((batch_size, 1)).to(torch.float).view(-1, 1).cuda()
            Tz = Tz.clamp(v_min, v_max)
            b = (Tz - v_min) / delta
            l = b.floor().to(torch.int64)
            u = b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.opt.num_outcomes - 1)) * (l == u)] += 1
            offset = torch.linspace(0, (batch_size - 1) * self.opt.num_outcomes, batch_size).to(torch.int64).unsqueeze(dim=1).expand(batch_size, self.opt.num_outcomes).cuda()
            skewed_anchor = torch.zeros(batch_size, self.opt.num_outcomes).cuda()
            skewed_anchor.view(-1).index_add_(0, (l + offset).view(-1), (target_anchor * (u.float() - b)).view(-1))  
            skewed_anchor.view(-1).index_add_(0, (u + offset).view(-1), (target_anchor * (b - l.float())).view(-1))  

            loss = -(skewed_anchor * (feat + 1e-16).log()).sum(-1).mean()

        elif self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True, feat_real=None):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator, feat_real)

                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator, feat_real)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


