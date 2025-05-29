import math

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, einsum
from torch.autograd import Variable
from torch import linalg as LA

from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from einops import rearrange, repeat

from utils import set_parameter_requires_grad, get_vector_property

from model.modules import import_class, bn_init, EncodingBlock, conv_init


class InfoGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, num_head=3, head=['ViT-B/32'], noise_ratio=0.1, k=0, gain=1):
        super(InfoGCN, self).__init__()

        A = np.stack([np.eye(num_point)] * num_head, axis=0)

        base_channel = 64
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * base_channel * num_point)
        self.noise_ratio = noise_ratio
        self.z_prior = torch.empty(num_class, base_channel*4)
        self.A_vector = self.get_A(graph, k)
        self.gain = gain
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))

        self.l1 = EncodingBlock(base_channel, base_channel,A)
        self.l2 = EncodingBlock(base_channel, base_channel,A)
        self.l3 = EncodingBlock(base_channel, base_channel,A)
        self.l4 = EncodingBlock(base_channel, base_channel*2, A, stride=2)
        self.l5 = EncodingBlock(base_channel*2, base_channel*2, A)
        self.l6 = EncodingBlock(base_channel*2, base_channel*2, A)
        self.l7 = EncodingBlock(base_channel*2, base_channel*4, A, stride=2)
        self.l8 = EncodingBlock(base_channel*4, base_channel*4, A)
        self.l9 = EncodingBlock(base_channel*4, base_channel*4, A)

        # add GAP
        #self.linear_head = nn.ModuleDict()
        # 定义一个形状为 (1, 5) 的可学习参数 self.logit_scale
        self.logit_scale = nn.Parameter(torch.ones(1,5) * np.log(1 / 0.07))
        # self.part_list = nn.ModuleList()
        # for i in range(4):
        #     self.part_list.append(nn.Linear(256,512))
        # self.head = head
        # if 'ViT-B/32' in self.head:
        #     self.linear_head['ViT-B/32'] = nn.Linear(256,512)
        #     conv_init(self.linear_head['ViT-B/32'])


        self.fc = nn.Linear(base_channel*4, base_channel*4)
        self.fc_mu = nn.Linear(base_channel*4, base_channel*4)
        self.fc_logvar = nn.Linear(base_channel*4, base_channel*4)
        self.decoder = nn.Linear(base_channel*4, num_class)
        nn.init.orthogonal_(self.z_prior, gain=gain)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def latent_sample(self, mu, logvar):
        '''
           mu: 均值向量，表示隐变量的期望值。
           logvar: 方差的对数向量，表示隐变量的方差的对数形式
        '''
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            # 将 logvar 乘以 self.noise_ratio将对数方差转换为标准差。
            # 因为方差是 exp(logvar)，所以标准差是 exp(logvar/2)，
            # 这里直接使用 exp() 是因为后续会与 eps 相乘，相当于隐式地处理了平方根。
            # 为了防止标准差过大，代码中使用了 torch.clamp(std, max=100) 来限制其最大值为 100。
            # std = logvar.exp()
            std = torch.clamp(std, max=100)
            # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x):
        '''
           N: batch_size
           C: feature dimension
           T: total frames
           V: number of joint
           M: num of person
        '''
        N, C, T, V, M = x.size()  # [128, 3, 64, 25, 2]
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous() # [16384, 25, 3]
        x = self.A_vector.to(x.device).float().expand(N*M*T, -1, -1) @ x # [16384, 25, 3]

        x = self.to_joint_embedding(x) # [16384, 25, 3]
        x += self.pos_embedding[:, :self.num_point] # [16384, 25, 3]
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous() # [128, 3200, 64]

        x = self.data_bn(x) # [128, 3200, 64] 
        x = rearrange(x, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous() # [256, 64, 64, 25]
        x = self.l1(x) # [256, 64, 64, 25]
        x = self.l2(x) # [256, 64, 64, 25]
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x) # [256, 256, 16, 25]

        # N*M,C,T,V
        c_new = x.size(1) # 256

        # add GAP, 4 parts
        feature = x.view(N,M,c_new,T//4,V)
        head_list = torch.Tensor([2,3,20]).long()
        hand_list = torch.Tensor([4,5,6,7,8,9,10,11,21,22,23,24]).long()
        foot_list = torch.Tensor([12,13,14,15,16,17,18,19]).long()
        hip_list = torch.Tensor([0,1,2,12,16]).long()
        head_feature = feature[:,:,:,:,head_list].mean(4).mean(3).mean(1)
        hand_feature = feature[:,:,:,:,hand_list].mean(4).mean(3).mean(1)
        foot_feature = feature[:,:,:,:,foot_list].mean(4).mean(3).mean(1)
        hip_feature = feature[:,:,:,:,hip_list].mean(4).mean(3).mean(1)

        x = x.view(N, M, c_new, -1) # [128, 2, 256, 400]
        x = x.mean(3).mean(1) # [128, 256]

        # #  Global Contrastive
        # for name in self.head:
        #     feature_dict[name] = self.linear_head[name](x)



        x = F.relu(self.fc(x))
        x = self.drop_out(x)
        z_mu = self.fc_mu(x) # [128, 256]
        z_logvar = self.fc_logvar(x) # [128, 256]
        z = self.latent_sample(z_mu, z_logvar)

        y_hat = self.decoder(z)

        return y_hat, z,  x, self.logit_scale, [head_feature, hand_feature, hip_feature, foot_feature]


class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model.float()

    def forward(self,text):
        return self.model.encode_text(text)
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        nn.init.xavier_uniform_(m.weight, gain=0.5) # gain = nn.init.calculate_gain('relu')

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self, layer_sizes, style_latent_size=0):
        super(Encoder, self).__init__()

        layers = []
        for i in range(len(layer_sizes)-2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.Dropout1d())
            layers.append(nn.ReLU())

        self.style_latent_size = style_latent_size

        self.model = nn.Sequential(*layers)
        self.mu = nn.Sequential(
            nn.Linear(layer_sizes[-2], layer_sizes[-1])
        )
        self.logvar = nn.Sequential(
            nn.Linear(layer_sizes[-2], layer_sizes[-1])
        )

        self.apply(weights_init)

    def forward(self, x, instance_style=False):

        h = self.model(x)
        mu = self.mu(h)
        logvar = self.logvar(h)

        if self.style_latent_size == 0:
            return mu, logvar

        if not instance_style:
            return (
                mu[:, :-self.style_latent_size],
                logvar[:, :-self.style_latent_size]
            )
        else:
            return (
                mu[:, :-self.style_latent_size],
                logvar[:, :-self.style_latent_size],
                mu[:, -self.style_latent_size:],
                logvar[:, -self.style_latent_size:]
            )


class Decoder(nn.Module):
    def __init__(self, layer_sizes):
        super(Decoder, self).__init__()

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

        self.apply(weights_init)

    def forward(self, x):

        out = self.model(x)
        return out
