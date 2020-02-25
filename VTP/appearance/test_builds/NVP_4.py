import argparse
import math
import torch
import time
from torch import nn, optim
from torch.nn import functional as F
from torch.nn import AvgPool2d
from torchvision import datasets, transforms
from torchsummary import summary
import sys,os
from VTP.utils.nn import spatial_broadcast_decoder, Residual


class VAE(nn.Module):
    def __init__(self, input_length, lsdim, device, logvar_bound):
        super(VAE, self).__init__()

        self.input_length = input_length
        self.lsdim = lsdim
        self.device = device
        self.logvar_bound = logvar_bound

        #Encoding Layers
        self.conv1A = nn.Conv2d(1, 64, 5,padding=2)
        self.conv1B = nn.Conv2d(64, 64, 5,padding=2)
        self.conv1C = nn.Conv2d(64, 64, 5,padding=2)
        self.conv1D = nn.Conv2d(64, 1, 5,padding=2)

        self.conv2A = nn.Conv2d(64, 64, 5,padding=2)
        self.conv2B = nn.Conv2d(64, 64, 5,padding=2)
        self.conv2C = nn.Conv2d(64, 64, 5,padding=2)
        self.conv2D = nn.Conv2d(64, 1, 5,padding=2)

        self.conv3A = nn.Conv2d(64, 64, 5,padding=2)
        self.conv3B = nn.Conv2d(64, 64, 5,padding=2)
        self.conv3C = nn.Conv2d(64, 64, 5,padding=2)
        self.conv3D = nn.Conv2d(64, 1, 5,padding=2)

        self.conv4A = nn.Conv2d(64, 64, 5,padding=2)
        self.conv4B = nn.Conv2d(64, 64, 5,padding=2)
        self.conv4C = nn.Conv2d(64, 64, 5,padding=2)
        self.conv4D = nn.Conv2d(64, 1, 5,padding=2)

        self.pw_conv0 = nn.Conv2d(5, 32, 1)

        self.res1_x = nn.ModuleList([Residual(pixelwise_channel=32, conv_channel=16,stride=2 if k==0 else 1) for k in range(3)])
        self.pw_conv1 = nn.Conv2d(32, 64, 1)
        self.res2_x = nn.ModuleList([Residual(pixelwise_channel=64, conv_channel=32,stride=2 if k==0 else 1) for k in range(2)])
        self.pw_conv2 = nn.Conv2d(64, 256, 1)
        self.res3_x = nn.ModuleList([Residual(pixelwise_channel=256, conv_channel=128,stride=2 if k==0 else 1) for k in range(1)])
        self.pw_conv3 = nn.Conv2d(256, 1024, 1)
        self.res4_x = nn.ModuleList([Residual(pixelwise_channel=1024, conv_channel=256,stride=2 if k==0 else 1) for k in range(1)])


        self.fcc=nn.Linear(1024,1000)

        self.mean = nn.Linear(1000, lsdim)
        self.logvar = nn.Linear(1000, lsdim)

        self.sbd=spatial_broadcast_decoder(input_length=self.input_length,device=self.device,lsdim=self.lsdim,channels=[64,64,64,64,64])
        self.res_d = nn.ModuleList([Residual(pixelwise_channel=64, conv_channel=32) for k in range(10)])
        self.pw_convd = nn.Conv2d(64, 1, 1)
        #Create an idle input for calling pseudo-inputs

    def reconstruct_x(self, x):
        x_mean, _, _, _ = self.forward(x)
        return x_mean

    def encoder(self, x):
        y1=x.clone()

        x=F.leaky_relu(self.conv1A(x))
        x=F.leaky_relu(self.conv1B(x))
        x=F.leaky_relu(F.max_pool2d(self.conv1C(x),(2,2)))
        y2=F.leaky_relu(self.conv1D(x))

        x=F.leaky_relu(self.conv2A(x))
        x=F.leaky_relu(self.conv2B(x))
        x=F.leaky_relu(F.max_pool2d(self.conv2C(x),(2,2)))
        y3=F.leaky_relu(self.conv2D(x))

        x=F.leaky_relu(self.conv3A(x))
        x=F.leaky_relu(self.conv3B(x))
        x=F.leaky_relu(F.max_pool2d(self.conv3C(x),(2,2)))
        y4=F.leaky_relu(self.conv3D(x))

        x=F.leaky_relu(self.conv4A(x))
        x=F.leaky_relu(self.conv4B(x))
        x=F.leaky_relu(F.max_pool2d(self.conv4C(x),(2,2)))
        x=F.leaky_relu(self.conv4D(x))

        y1=y1
        y2=F.upsample(y2,scale_factor=2)
        y3=F.upsample(y3,scale_factor=4)
        y4=F.upsample(y4,scale_factor=8)
        x=F.upsample(x,scale_factor=16)

        x=torch.cat((y1,y2,y3,y4,x),dim=1)

        x = F.leaky_relu(self.pw_conv0(x))

        for r in self.res1_x:
            x = r(x)

        x = F.leaky_relu(self.pw_conv1(x))

        for r in self.res2_x:
            x = r(x)

        x = F.leaky_relu(self.pw_conv2(x))

        for r in self.res3_x:
            x = r(x)

        x = F.leaky_relu(self.pw_conv3(x))

        for r in self.res4_x:
            x = r(x)

        x = F.avg_pool2d(x,(8,8))

        x=x.view(-1,1024)
        x=F.leaky_relu(self.fcc(x))

        z_mean = self.mean(x)
        z_logvar = F.elu(self.logvar(x), -1.*self.logvar_bound)
        return z_mean, z_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        x = self.sbd(z)

        for r in self.res_d:
            x = r(x)
        
        x = F.leaky_relu(self.pw_convd(x))

        return torch.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z=self.reparameterize(mu,logvar)
        x_mean=self.decode(z)
        return x_mean, mu, logvar, z

class PseudoGen(nn.Module):
    def __init__(self, input_length, pseudos,device):
        super(PseudoGen, self).__init__()

        self.means = nn.Linear(pseudos, input_length*input_length, bias=False)
        self.idle_input = torch.eye(pseudos,pseudos,requires_grad=True)
        self.idle_input = self.idle_input.to(device)

    def forward(self, x):
        return torch.sigmoid(self.means(x))

class NVP_4(nn.Module):
    def __init__(self, input_length, lsdim, pseudos, beta, gamma, device, logvar_bound):
        super(NVP_4, self).__init__()

        self.input_length = input_length
        self.pseudos = pseudos
        self.beta = beta
        self.gamma = gamma

        self.vae = VAE(input_length, lsdim, device, logvar_bound)
        self.pseudoGen = PseudoGen(input_length, pseudos,device)

        self.idle_input = torch.eye(pseudos, requires_grad=True).to(device)

    def forward(self, x):
        return self.vae.forward(x)

    def loss_function(self, recon_x, x, mu, logvar, z_q, pseudo,recon_pseudo, p_mu, p_logvar, p_z, gamma=None):
        reconstructionLoss = F.mse_loss(recon_x.view(-1,self.input_length*self.input_length), x.view(-1, self.input_length*self.input_length), reduction = 'sum')

        log_p_z = self.log_p_z(z_q)
        log_q_z = torch.sum(self.log_Normal_diag(z_q, mu, logvar, dim=1),0)
        KL = -(log_p_z - log_q_z)

        pseudoReconstructionLoss = F.mse_loss(recon_pseudo.view(-1,self.input_length*self.input_length), pseudo.view(-1, self.input_length*self.input_length), reduction = 'sum')

        plog_p_z = self.log_p_z(p_z)
        plog_q_z = torch.sum(self.log_Normal_diag(p_z, p_mu, p_logvar, dim=1),0)
        pKL= -(plog_p_z - plog_q_z)

        if gamma is None:
            gamma=self.gamma
        return (reconstructionLoss + self.beta*KL)+(x.shape[0] * gamma / self.pseudos)*(pseudoReconstructionLoss + self.beta*pKL)

    def bayesian_loss(self, recon_x, x, mu, logvar, z_q, pseudo,recon_pseudo, p_mu, p_logvar, p_z, gamma=None):
        reconstructionLoss = F.mse_loss(recon_x.view(-1,self.input_length*self.input_length), x.view(-1, self.input_length*self.input_length), reduction = 'sum')

        log_p_z = self.log_p_z(z_q)
        log_q_z = torch.sum(self.log_Normal_diag(z_q, mu, logvar, dim=1),0)
        KL = -(log_p_z - log_q_z)

        pseudoReconstructionLoss = F.mse_loss(recon_pseudo.view(-1,self.input_length*self.input_length), pseudo.view(-1, self.input_length*self.input_length), reduction = 'sum')

        plog_p_z = self.log_p_z(p_z)
        plog_q_z = torch.sum(self.log_Normal_diag(p_z, p_mu, p_logvar, dim=1),0)
        pKL= -(plog_p_z - plog_q_z)

        if gamma is None:
            gamma=self.gamma
        lossSum = (reconstructionLoss + self.beta*KL)+(gamma / self.pseudos)*(pseudoReconstructionLoss + self.beta*pKL)
        lossAverage = lossSum / (x.shape[0] + gamma)
        return x.shape[0] * lossAverage

    def log_Normal_diag(self, x, mean, log_var, average=False, dim=None):
        #print(log_var)
        #T:(batch-size, num-pseudos, lsdim)
        #T[i,j,k]=element i, marginal probability along axis k for posterior j
        log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )

        #T: (batch-size, num-pseudos)
        #T[i,j]=log probability that element i originates from posterior j
        if average:

            return torch.mean( log_normal, dim )

        else:

            return torch.sum( log_normal, dim )

    def log_p_z(self,z):
        # calculate params
        X = self.pseudoGen.forward(self.idle_input)

        # calculate params for given data
        z_p_mean, z_p_logvar = self.vae.encoder(X.view(-1,1,self.input_length,self.input_length))  # C x M

        #INCLUDE LATEX WRITEUP
        z_expand = z.unsqueeze(1)
        #z_expand = z_expand.repeat(1, self.pseudos, 1)

        means = z_p_mean.unsqueeze(0)
        #means = means.repeat(z_expand.shape[0], 1, 1)

        logvars = z_p_logvar.unsqueeze(0)
        #logvars= means.repeat(z_expand.shape[0], 1, 1)

        a = self.log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(self.pseudos)  # MB x C

        #We will scale by the maximum posterior probabilities in order to ensure numerical stability in later stages
        a_max, _ = torch.max(a, 1)  # MB x 1

        # calculate log-sum-exp
        #Use of a_max cancels out but is kept for numerical stability of exp operation
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB x 1
        return torch.sum(log_prior, 0)
