import torch
from torch import nn, eye
from torch.nn import functional as F
from torch.nn import LeakyReLU
from scipy.ndimage import gaussian_filter


'''
Gaussian Filter Layer

@author Meekail Zain
'''
class GaussianLayer(nn.Module):
    '''
    Constructs a gaussian filter layer leveraging
    scipy's ndimage gaussian filter functionality
    
    @param channels_in number of input channels
    @param channels_out number of output channels
    @param kernel_size size of a kernel
    @param sigma sigma value for gaussian filter, 
    or how much to 'blur' the image
    '''
    
    def __init__(self,channels_in,channels_out,kernel_size,sigma):
        super(GaussianLayer, self).__init__()
        assert kernel_size%2 == 1, "please choose an odd kernel size"
        self.r=(kernel_size-1)//2
        self.sigma=sigma
        self.kernel_size=kernel_size
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.r), 
            nn.Conv2d(channels_in, channels_out, kernel_size, bias=None, groups=channels_in)
        )

        self.weights_init()
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((self.kernel_size,self.kernel_size))
        n[self.r,self.r] = 1
        k = gaussian_filter(n,sigma=self.sigma,truncate=self.r if self.r < 6*self.sigma else 6*self.sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))

'''
Depthwise-separable convolutional layer

@author Meekail Zain
'''
class depthwise_separable_conv(nn.Module):
    '''
    Constructs a depthwise-separable convolutional layer
    
    @param nin number of input channels
    @param nout number of output channels
    @param kpl kernels per layer
    @param kernel_size size of a kernel
    @param padding amount of zero padding
    '''
    def __init__(self, nin, nout, kpl=1, kernel_size=3, padding=0):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kpl, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin * kpl, nout, kernel_size=1)

    '''
    Applies the layer to an input x
    
    @param x the input
    @return the layer's output
    '''
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

'''
Spatial broadcast decoder for images with equal width and height

@author Meekail Zain
'''
class spatial_broadcast_decoder(nn.Module):
    '''
    Constructs spatial broadcast decoder
    
    @param input_length width of image
    @param device torch device for computations
    @param lsdim dimensionality of latent space
    @param kernel_size size of size-preserving kernel. Must be odd.
    @param channels list of output-channels for each of the four size-preserving convolutional layers
    '''
    def __init__(self,input_length,device,lsdim,kernel_size=3,channels=[64,64,64,64]):
        super(spatial_broadcast_decoder,self).__init__()
        self.input_length=input_length
        self.device=device
        self.lsdim=lsdim
        assert kernel_size%2==1, "Kernel size must be odd"
        padding=int((kernel_size-1)/2)
        #Size-Preserving Convolutions
        self.conv1 = nn.Conv2d(lsdim + 2, channels[0], kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv2d(channels[3], 1, 1)

    '''
    Applies the spatial broadcast decoder to a code z
    
    @param z the code to be decoded
    @return the decoding of z
    '''
    def forward(self,z):
        baseVector = z.view(-1, self.lsdim, 1, 1)
        base = baseVector.repeat(1,1,self.input_length,self.input_length)

        stepTensor = torch.linspace(-1, 1, self.input_length).to(self.device)

        xAxisVector = stepTensor.view(1,1,self.input_length,1)
        yAxisVector = stepTensor.view(1,1,1,self.input_length)

        xPlane = xAxisVector.repeat(z.shape[0],1,1,self.input_length)
        yPlane = yAxisVector.repeat(z.shape[0],1,self.input_length,1)

        base = torch.cat((xPlane, yPlane, base), 1)         

        x = F.leaky_relu(self.conv1(base))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        return x
        
'''
Spatial Broadcast Decoder for images of equal width and height with Batch Normalization
Note that batch normalization is unstable as of 9/25/2019

@param input_length 
@author Quinn Wyner
'''
class spatial_broadcast_decoder_batchnorm(nn.Module):
    '''
    Constructs spatial broadcast decoder
    
    @param input_length width of image
    @param device torch device for computations
    @param lsdim dimensionality of latent space
    @param kernel_size size of size-preserving kernel. Must be odd.
    @param channels list of output-channels for each of the four size-preserving convolutional layers
    '''
    def __init__(self,input_length,device,lsdim,kernel_size=3,channels=[64,64,64,64]):
        super(spatial_broadcast_decoder_batchnorm,self).__init__()
        self.input_length=input_length
        self.device=device
        self.lsdim=lsdim
        assert kernel_size%2==1, "Kernel size must be odd"
        padding=int((kernel_size-1)/2)
        
        #Size-Preserving Convolutions
        
        #(lsdim + 2, input_length, input_length) -> (channels[0], input_length, input_length)
        self.conv1 = nn.Conv2d(lsdim + 2, channels[0], kernel_size=kernel_size, padding=padding)
        
        #(channels[0], input_length, input_length) -> (channels[0], input_length, input_length)
        self.batch1 = nn.BatchNorm2d(channels[0])
        
        #(channels[0], input_length, input_length) -> (channels[1], input_length, input_length)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, padding=padding)
        
        #(channels[1], input_length, input_length) -> (channels[1], input_length, input_length)
        self.batch2 = nn.BatchNorm2d(channels[1])
        
        #(channels[1], input_length, input_length) -> (channels[2], input_length, input_length)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size, padding=padding)
        
        #(channels[2], input_length, input_length) -> (channels[2], input_length, input_length)
        self.batch3 = nn.BatchNorm2d(channels[2])
        
        #(channels[2], input_length, input_length) -> (channels[3], input_length, input_length)
        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernel_size, padding=padding)
        
        #(channels[3], input_length, input_length) -> (channels[3], input_length, input_length)
        self.batch4 = nn.BatchNorm2d(channels[3])
        
        #(channels[3], input_length, input_length) -> (1, input_length, input_length)
        self.conv5 = nn.Conv2d(channels[3], 1, 1)
        
        #(1, input_length, input_length) -> (1, input_length, input_length)
        self.batch5 = nn.BatchNorm2d(1)

    '''
    Applies the spatial broadcast decoder to a code z
    
    @param z the code to be decoded
    @return the decoding of z
    '''
    def forward(self,z):
        baseVector = z.view(-1, self.lsdim, 1, 1)
        base = baseVector.repeat(1,1,self.input_length,self.input_length)

        stepTensor = torch.linspace(-1, 1, self.input_length).to(self.device)

        xAxisVector = stepTensor.view(1,1,self.input_length,1)
        yAxisVector = stepTensor.view(1,1,1,self.input_length)

        xPlane = xAxisVector.repeat(z.shape[0],1,1,self.input_length)
        yPlane = yAxisVector.repeat(z.shape[0],1,self.input_length,1)

        base = torch.cat((xPlane, yPlane, base), 1)         

        #(lsdim+2, input_length, input_length) -> (channels[0], input_length, input_length)
        x = F.leaky_relu(self.batch1(self.conv1(base)))
        
        #(channels[0], input_length, input_length) -> (channels[1], input_length, input_length)
        x = F.leaky_relu(self.batch2(self.conv2(x)))
        
        #(channels[1], input_length, input_length) -> (channels[2], input_length, input_length)
        x = F.leaky_relu(self.batch3(self.conv3(x)))
        
        #(channels[2], input_length, input_length) -> (channels[3], input_length, input_length)
        x = F.leaky_relu(self.batch4(self.conv4(x)))
        
        #(channels[3], input_length, input_length) -> (1, input_length, input_length)
        x = F.leaky_relu(self.batch5(self.conv5(x)))

        return x
        
'''
Spatial broadcast decoder for use on images that do not necessarily have the same height and width

@author Quinn Wyner
'''
class spatial_broadcast_decoder_asymmetric(nn.Module):
    '''
    Constructs a spatial broadcast decoder
    
    @param input_height height of an input image
    @param input_width width of an input image
    @param device torch device for computations
    @param lsdim dimensionality of the latent space
    @param kernel_size size of a convolutional layer's kernel
    @param channels list of output channels for each convolutional layer
    '''
    def __init__(self,input_height,input_width,device,lsdim,kernel_size=3,channels=[64,64,64,64]):
        super(spatial_broadcast_decoder_asymmetric,self).__init__()
        self.input_height = input_height
        self.input_width=input_width
        self.device=device
        self.lsdim=lsdim
        assert kernel_size%2==1, "Kernel size must be odd"
        padding=int((kernel_size-1)/2)
        #Size-Preserving Convolutions
        self.conv1 = nn.Conv2d(lsdim + 2, channels[0], kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv2d(channels[3], 1, 1)

    '''
    Decodes a given code z
    
    @param z code to be decoded
    @return the decoding of z
    '''
    def forward(self,z):
        baseVector = z.view(-1, self.lsdim, 1, 1)
        base = baseVector.repeat(1,1,self.input_height,self.input_width)

        heightStepTensor = torch.linspace(-1, 1, self.input_height).to(self.device)
        widthStepTensor = torch.linspace(-1, 1, self.input_width).to(self.device)
        heightAxisVector = heightStepTensor.view(1,1,self.input_height,1)
        widthAxisVector = widthStepTensor.view(1,1,1,self.input_width)

        xPlane = heightAxisVector.repeat(z.shape[0],1,1,self.input_width)
        yPlane = widthAxisVector.repeat(z.shape[0],1,self.input_height,1)

        base = torch.cat((xPlane, yPlane, base), 1)         

        x = F.leaky_relu(self.conv1(base))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))

        return x

"""
ResNet 'deeper' block as used in ResNet-110

@author Meekail Zain
"""
class Residual(nn.Module):
    def __init__(self,pixelwise_channel=256,conv_channel=64,kernel_size=3):
        super(ResNetBlock, self).__init__()
        self.conv1=nn.Conv2d(pixelwise_channel,conv_channel,1)
        self.conv2=nn.Conv2d(conv_channel,conv_channel,kernel_size)
        self.conv3=nn.Conv2d(conv_channel,pixelwise_channel,1)

    '''
    Applies the ResNetBlock to an input x
    
    @param x the input
    @return the output of the ResNetBlock
    '''
    def forward(self, x):
        x_ = x.clone()
        x=F.leaky_relu(self.conv1())
        x=F.leaky_relu(self.conv2())
        x=F.leaky_relu(self.conv3())
        return x+x_
        
        
        
        

"""
ResNet-style block that preserves dimensionality of input

@author Quinn Wyner
"""
class ResNetBlock(nn.Module):
    
    '''
    Constructs a ResNetBlock
    
    @param channels number of filters
    @param kernel_size size of a kernel; either an int or tuple of 2 ints
    @param numLayers number of convolutions to perform
    @param activationFunction function to perform on layers; either a lambda function or a tuple of lambda functions
    @param shortcutInterval number of layers between each shortcut
    '''
    def __init__(self, channels, kernel_size, numLayers, activationFunction, shortcutInterval):
        super(ResNetBlock, self).__init__()
        if type(activationFunction) == tuple and len(activationFunction) != numLayers:
            raise Exception(f'length of activation function must be same as numLayers {numLayers}, but is instead {len(activationFunction)}')
        self.activationFunction = activationFunction
        self.shortcutInterval = shortcutInterval
        if type(kernel_size) == int:
            if kernel_size % 2 == 0:
                raise Exception(f'kernel_size must exclusively have odd values, but has value {kernel_size}')
                return
            self.layers = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2) for i in range(numLayers)])

        else:
            for i in range(2):
                if kernel_size[i] % 2 == 0:
                    raise Exception(f'kernel_size must exclusively have odd values, but has value {kernel_size[0]}')
                    return
            self.layers = nn.ModuleList([nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2)) for i in range(numLayers)])
            
    '''
    Applies the ResNetBlock to an input x
    
    @param x the input
    @return the output of the ResNetBlock
    '''
    def forward(self, x):
        shortcut = x
        z = x
        for i in range(len(self.layers)):
            shortcutLayer = ((i+1) % self.shortcutInterval == 0)
            z = self.layers[i](z)
            if shortcutLayer:
                z = z + shortcut
            if type(self.activationFunction) == tuple:
                z = self.activationFunction[i](z)
            else:
                z = self.activationFunction(z)
            if shortcutLayer:
                shortcut = z
        return z