import torch
from torch import nn
import torch.nn.functional as f
from torchsummary import summary

from VTP.dynamics.encoders.feedforward import FeedForward
from VTP.dynamics.decoders.hidden_state import HiddenState
from VTP.dynamics.decoders.neural_integrator import NeuralIntegrator
from VTP.dynamics.decoders.recurrent import Recurrent
from VTP.dynamics.decoders.sequential import Sequential

ENCODERS = {
    "feed forward": FeedForward
}

DECODERS = {
    "hidden state": HiddenState,
    "neural integrator": NeuralIntegrator,
    "recurrent": Recurrent,
    "sequential": Sequential
}

class VTP_D(nn.Module):
    def __init__(self, encoder_type="feedforward", decoder_type="hidden state"):
        super(VTP_D, self).__init__()
        self.encoder = ENCODERS[encoder_type]()
        self.decoder = DECODERS[decoder_type]()
     
    # sample using mu and logvar
    def sample_from(self, mu, logvar):
        eps = torch.randn_like(mu)
        std_dev = torch.exp(logvar / 2)
        return mu + eps * std_dev
           
    def forward(self, p):
        mu, logvar = self.encoder(p)
        w = self.sample_from(mu, logvar)
        return self.decoder(w)

#model = VTP_D()
#summary(model, (20, 2))
        
    
        
        
        
        
        
        
        
        
