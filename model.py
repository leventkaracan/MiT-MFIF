import torch
import torch.nn as nn
from core.spectral_norm import spectral_norm as _spectral_norm
from torch.nn import functional as F
import math
import numpy as np
from torchvision import transforms as T
from torchvision.utils import save_image
import time

    
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

############################## Discriminator  ##################################

def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module



class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=True):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=spectral_norm, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.ReLU(True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                #nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                #norm_layer(nf), nn.ReLU(True)
                spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw, bias=False), True),
                #nn.ReLU(True)
                nn.LeakyReLU(0.1, True)
                
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            #nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            #norm_layer(nf),
            #nn.ReLU(True)
            spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), True),
            #nn.ReLU(True)
            nn.LeakyReLU(0.1, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
            
############################## Focus Map Generator  ##################################


class FocusMapGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(FocusMapGenerator, self).__init__()

        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True)
            )
           
        channel = 256
        stack_num = 8
        blocks = []
        for _ in range(stack_num):
            blocks.append(MultiimageTransformer(hidden=channel))
        self.transformer = nn.Sequential(*blocks)
       
        # decoder: decode frames from features       
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channel*2, 128, kernel_size=4, stride=2, padding=1,  bias=False),
            nn.BatchNorm2d(128, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
            )
            
        if init_weights:
            self.init_weights()
            

    def forward(self, imgA, imgB):
        
        t = 2
        img_AB = torch.cat((imgA.unsqueeze(0), imgB.unsqueeze(0)),dim=0)
        img_AB = img_AB.permute(1, 0, 2, 3, 4)
        b, t, c, h, w = img_AB.size()

        enc_feat = self.encoder(img_AB.contiguous().view(b*t, c, h, w))
        _, c, h, w = enc_feat.size()
   
        enc_feat = self.transformer(
            {'x': enc_feat, 'b': b, 'c': c})['x']
        enc_feat = enc_feat.view(b,t,c,h,w)
        enc_feat = torch.cat((enc_feat[:,0], enc_feat[:,1]), 1)
 
        output = self.decoder(enc_feat)
        # If the sizes of input images are not multiples of two, uncomment the following line.
        #output = F.interpolate(output, (imgA.size(2),imgA.size(3)), mode='nearest')
 
        return output


# ############################# Transformer  ##################################
# #############################################################################

class Attention(nn.Module):


    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn

class MultiHead(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
            #nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()
        
    def calculate_patchsize(self,h,w,l):
        patchsizes = []

        divval= np.power(2,l-1)

        h_min_ps = int(np.ceil(h/divval))
        

        w_min_ps = int(np.ceil(w/divval))
        patchsizes.append((h_min_ps, w_min_ps))
        h_ps = h_min_ps
        w_ps = w_min_ps
        for i in range(1,l):
            h_ps = int(h_ps * 2)
            w_ps = int(w_ps * 2)
            patchsizes.append((h_ps, w_ps))
        return patchsizes[::-1], (divval * h_min_ps, divval * w_min_ps)

    def forward(self, x, b, c):
        
        bt, _, h, w = x.size()
        patchsize, hw = self.calculate_patchsize(h,w,4)

        hpad1 = 0
        wpad1 = 0
        hpad2 = 0
        wpad2 = 0

        if(hw[0]>h or hw[1]>w):
            hpad1 = int(np.floor((hw[0] - h)/2))
            hpad2 = int(np.ceil((hw[0] - h)/2))

            
            wpad1 = int(np.floor((hw[1] - w)/2))
            wpad2 = int(np.ceil((hw[1] - w)/2))

            x = F.pad(x, (wpad1,wpad2,hpad1,hpad2),"reflect")
            

        bt, _, h, w = x.size()

        t = bt // b
        d_k = c // len(patchsize)
        output = []

        _key = self.key_embedding(x)
        _query = self.query_embedding(x)
        _value = self.value_embedding(x)
        for (height, width), query, key, value in zip(patchsize,
                                                      torch.chunk(_query, len(patchsize), dim=1), torch.chunk(
                                                          _key, len(patchsize), dim=1),
                                                      torch.chunk(_value, len(patchsize), dim=1)):

            out_w, out_h = w // width, h // height
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)

            y, _ = self.attention(query, key, value)
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
 

        if hpad1>0 or wpad1>0 or hpad2>0 or wpad2>0 :
            if (hpad1>0 or hpad2>0) and (wpad1>0 or wpad2>0):
                output =  output[:,:,hpad1:-hpad2, wpad1:-wpad2]
            elif(hpad1>0 or hpad2>0):
                output =  output[:,:,hpad1:-hpad2, :]
            else:
                output =  output[:,:,:, wpad1:-wpad2]

        x = self.output_linear(output)
        return x



        
        
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
        
        
        
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel, bias=False),
            nn.Sigmoid()
            #h_sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y.expand_as(x) 


class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='se', reduction=16,
                 wo_dp_conv=False, dp_first=False):
        """
        We build our FeedForward network on LocalViT to ensure locality in proposed Multi-image Transformer
        "Li, Y., Zhang, K., Cao, J., Timofte, R., & Van Gool, L. (2021). 
        Localvit: Bringing locality to vision transformers. 
        arXiv preprint arXiv:2104.05707."
        """
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        # the first linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU6(inplace=True)])

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.GroupNorm(1, hidden_dim),
                nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)


        layers.append(SELayer(hidden_dim, reduction=reduction))


        # the second linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.GroupNorm(1, out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x



class MultiimageTransformer(nn.Module):
    """
    Multi-image Transformer(MiT)
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.attention = MultiHead(d_model=hidden)
        self.feed_forward = LocalityFeedForward(hidden, hidden, 1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x, b, c = x['x'], x['b'], x['c']
        x = x + self.attention(x, b, c)
        x = self.feed_forward(x)
        x = self.relu(x)

        return {'x': x, 'b': b, 'c': c}
        
