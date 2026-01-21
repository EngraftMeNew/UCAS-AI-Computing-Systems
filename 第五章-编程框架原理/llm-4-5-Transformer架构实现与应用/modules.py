# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from torch.nn.parameter import Parameter
from hyperparams import *

class embedding(nn.Module):

    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        '''Embeds a given Variable.
        Args:
          vocab_size: An int. Vocabulary size.
          num_units: An int. Number of embedding hidden units.
          zero_pad: A boolean. If True, all the values of the fist row (id 0)
            should be constant zeros.
          scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
        '''
        super(embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = Parameter(torch.Tensor(vocab_size, num_units))
        #TODO：使用Xavier正态初始化方法对嵌入矩阵进行初始化
        nn.init.xavier_normal_(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        #TODO：调用内置函数将输入的词索引映射为对应的词向量表示，用于实现嵌入层功能
        outputs = F.embedding(
            inputs, self.lookup_table, self.padding_idx, None, 2, False, False)

        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class layer_normalization(nn.Module):

    def __init__(self, features, epsilon=1e-8):
        '''Applies layer normalization.

        Args:
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        '''
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        #TODO：定义可训练的缩放参数 gamma，初始值全为1，形状与features维度一致
        self.gamma = nn.Parameter(torch.ones(features))
        #TODO：定义可训练的偏移参数 beta，初始值全为0，形状与features维度一致
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        #TODO：对输入计算最后一个维度的均值
        mean = x.mean(-1, keepdim=True)
        #TODO：对输入计算最后一个维度的标准差
        std = x.std(-1, keepdim=True)
        #TODO：对输入进行层归一化
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class positional_encoding(nn.Module):

    def __init__(self, num_units, zeros_pad=True, scale=True):
        '''Sinusoidal Positional_Encoding.

        Args:
          num_units: Output dimensionality
          zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
          scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
        '''
        super(positional_encoding, self).__init__()
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale

    def forward(self, inputs, y):
        # inputs: A 2d Tensor with shape of (N, T).
        N, T = inputs.size()[0: 2]

        # First part of the PE function: sin and cos argument
        # position_ind = Variable(torch.unsqueeze(torch.arange(0, T), 0).repeat(N, 1).cuda().long())
        position_ind = torch.unsqueeze(torch.arange(0, T), 0).repeat(N, 1)
        if (inputs.is_cuda):
            position_ind = Variable(position_ind.cuda().long())
        if (inputs.device.type == 'mlu'):
            position_ind = Variable(position_ind.to('mlu').long())


        #TODO: 根据论文公式，计算位置编码矩阵，形状为(T, num_units)
        position_enc = torch.Tensor(np.array([
            [pos / np.power(10000, 2. * i / self.num_units) for i in range(self.num_units)]
            for pos in range(T)
        ]))

        # Second part, apply the cosine to even columns and sin to odds.
        #TODO：对偶数列（从0开始）使用sin
        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])  # dim 2i
        #TODO：对奇数列使用cos
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2]) # dim 2i+1

        # Convert to a Variable
        # lookup_table = Variable(position_enc).cuda()
        lookup_table = Variable(position_enc)
        if (inputs.is_cuda):
            lookup_table = lookup_table.cuda()
        if (inputs.device.type == 'mlu'):
            lookup_table = lookup_table.to('mlu').to(y.dtype)


        if self.zeros_pad:
            lookup_table = torch.cat((Variable(torch.zeros(1, self.num_units)),
                                     lookup_table[1:, :]), 0)
            padding_idx = 0
        else:
            padding_idx = -1

        #TODO: 根据位置索引 position_ind，从位置编码查找表 lookup_table 中取出对应的位置编码向量，生成最终的编码输出。
        outputs = F.embedding(
            position_ind, lookup_table, padding_idx, None, 2, False, False)

        if self.scale:
            #TODO：将输出进行缩放
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class multihead_attention(nn.Module):

    def __init__(self, hp_, num_units, num_heads=8, dropout_rate=0, causality=False):
        '''Applies multihead attention.

        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            causality: Boolean. If true, units that reference the future are masked.
            num_heads: An int. Number of heads.
        '''
        super(multihead_attention, self).__init__()
        self.hp = hp_
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        #TODO：构建Q、K、V的线性映射层，包含全连接和ReLU激活，用于将输入特征投影到注意力空间
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        #TODO：输出dropout层
        self.output_dropout = nn.Dropout(self.dropout_rate)
        #TODO：调用自定义函数实现层归一化，标准化输出
        self.normalization = layer_normalization(self.num_units)
        #self.normalization = nn.LayerNorm(self.num_units, eps=1e-8)

    def forward(self, queries, keys, values):
        # keys, values: same shape of [N, T_k, C_k]
        # queries: A 3d Variable with shape of [N, T_q, C_q]

        # Linear projections
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)

        # Split and concat
        #TODO：将Q, K, V按最后一维均分为num_heads份，并在batch维拼接
        Q_ = torch.cat(torch.split(Q, self.num_units // self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, self.num_units // self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.split(V, self.num_units // self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)

        # Multiplication
        #TODO：计算Q与K的转置在每个注意力头内的批量矩阵乘法，得到注意力得分
        outputs = torch.bmm(Q_, K_.transpose(1, 2))  # (h*N, T_q, T_k)

        # Scale
        #TODO：按键的最后一维度平方根缩放注意力得分
        outputs = outputs / (K_.size(-1) ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)

        #padding = Variable(torch.ones(*outputs.size()).cuda() * (-2 ** 32 + 1))
        init_tensor = torch.ones(*outputs.size(),dtype=queries.dtype)
        if (queries.is_cuda):
            init_tensor = init_tensor.cuda()
        if (queries.device.type == 'mlu'):
            init_tensor = init_tensor.to('mlu')
        padding = Variable(init_tensor * (-2 ** 32 + 1))

        condition = key_masks.eq(0.)
        outputs = padding * condition + outputs * (~condition) #pytorch1.2 version incompatibility, change 1.-b to ~b

        # Causality = Future blinding
        if self.causality:
            #diag_vals = torch.ones(*outputs[0, :, :].size()).cuda()  # (T_q, T_k)
            diag_vals = torch.ones(*outputs[0, :, :].size(),dtype=queries.dtype)  # (T_q, T_k)
            if (queries.is_cuda):
                diag_vals = diag_vals.cuda()
            if (queries.device.type == 'mlu'):
                diag_vals = diag_vals.to('mlu')

            #TODO：生成一个下三角矩阵，主对角线及其以下由 diag_vals 填充，其余位置为零
            tril = torch.tril(diag_vals)  # (T_q, T_k)
            # print(tril)
            masks = Variable(torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1))  # (h*N, T_q, T_k)

            #padding = Variable(torch.ones(*masks.size()).cuda() * (-2 ** 32 + 1))
            mask = torch.ones(*masks.size(),dtype=queries.dtype)
            if (queries.is_cuda):
                mask = mask.cuda()
            if (queries.device.type == 'mlu'):
                mask = mask.to('mlu')
            #TODO：生成一个下三角矩阵，主对角线及其以下由 diag_vals 填充，其余位置为零
            tril = torch.tril(diag_vals)  # (T_q, T_k)
            padding = Variable(mask * (-2 ** 32 + 1))

            condition = masks.eq(0.)
            outputs = padding * condition + outputs * (~condition) #pytorch1.2 version incompatibility, change 1.-b to ~b

        # Activation
        #TODO：对最后一个维度做softmax，计算注意力权重
        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        #TODO：屏蔽无效的 query 位置，防止其影响注意力结果
        outputs = outputs * query_masks

        # Dropouts
        #TODO：对注意力权重做dropout
        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)

        # Weighted sum
        #TODO：执行批量矩阵乘法，将注意力权重与值向量相乘
        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        # Restore shape
        #TODO：将多头的输出按特征维度拼接，还原为(N, T_q, C)
        outputs = torch.cat(torch.split(outputs, queries.size(0), dim=0), dim=2)  # (N, T_q, C)

        # Residual connection
        #TODO： 加入残差连接
        outputs += queries

        # Normalize
        #TODO：对输出进行层归一化
        outputs = self.normalization(outputs)  # (N, T_q, C)

        return outputs


class feedforward(nn.Module):

    def __init__(self, in_channels, num_units=[2048, 512]):
        '''Point-wise feed forward net.

        Args:
          in_channels: a number of channels of inputs
          num_units: A list of two integers.
        '''
        super(feedforward, self).__init__()
        self.in_channels = in_channels
        self.num_units = num_units

        # nn.Linear is faster than nn.Conv1d
        self.conv = False
        if self.conv:
            params = {'in_channels': self.in_channels, 'out_channels': self.num_units[0],
                      'kernel_size': 1, 'stride': 1, 'bias': True}
            #TODO：采用卷积构建第一层线性映射，（构建包含一维卷积和ReLU激活的顺序网络，卷积参数由字典params解包传入）
            self.conv1 = nn.Sequential(nn.Conv1d(**params), nn.ReLU())
            params = {'in_channels': self.num_units[0], 'out_channels': self.num_units[1],
                      'kernel_size': 1, 'stride': 1, 'bias': True}
           #TODO：采用卷积构建第二层线性映射，（定义一个一维卷积层，卷积参数通过字典params解包传入）
            self.conv2 = nn.Conv1d(**params)
        else:
            #TODO：采用全连接方式实现第一层线性映射
            self.conv1 = nn.Sequential(
                nn.Linear(self.in_channels, self.num_units[0]),
                nn.ReLU()
            )
            #TODO：采用全连接方式实现第二层线性映射
            self.conv2 = nn.Linear(self.num_units[0], self.num_units[1])
        #TODO：调用自定义函数实现层归一化，标准化输出
        self.normalization = layer_normalization(self.in_channels)
        #self.normalization = nn.LayerNorm(self.in_channels, eps=1e-8)

    def forward(self, inputs):
        if self.conv:
            #TODO：调整输入形状(batch_size, seq_len, channels) -> (batch_size, channels, seq_len)
            inputs = inputs.transpose(1, 2)
        #TODO：构建第一层线性映射
        outputs = self.conv1(inputs)
        if not self.conv:
            outputs = F.relu(outputs)
        #TODO：构建第二层线性映射
        outputs = self.conv2(outputs)

        #TODO：残差连接
        outputs += inputs

        # Layer normalization
        #TODO： 如果是卷积实现，先进行形状转换再进行层归一化
        if self.conv:
            outputs = self.normalization(outputs.transpose(1, 2))
        #TODO：如果是线性映射，直接归一化
        else:
            outputs = self.normalization(outputs)

        return outputs


class label_smoothing(nn.Module):

    def __init__(self, epsilon=0.1):
        '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

        Args:
            epsilon: Smoothing rate.
        '''
        super(label_smoothing, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        #TODO： 获取类别数量
        K = inputs.size(-1)
        #TODO：应用公式进行标签平滑
        return (1 - self.epsilon) * inputs + self.epsilon / K
