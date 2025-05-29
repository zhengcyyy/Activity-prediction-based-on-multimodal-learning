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

from model.modules import import_class, bn_init, EncodingBlock


class InfoGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, num_head=3, noise_ratio=0.1, k=0, gain=1):
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
        x = x.view(N, M, c_new, -1) # [128, 2, 256, 400]
        x = x.mean(3).mean(1) # [128, 256]
        x = F.relu(self.fc(x))
        x = self.drop_out(x)

        z_mu = self.fc_mu(x) # [128, 256]
        z_logvar = self.fc_logvar(x) # [128, 256]
        z = self.latent_sample(z_mu, z_logvar)

        y_hat = self.decoder(z)

        return y_hat, z
