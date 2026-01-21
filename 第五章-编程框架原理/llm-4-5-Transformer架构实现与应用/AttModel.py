# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from modules import *
from hyperparams import *
from hyperparams import Hyperparams as hp

class AttModel(nn.Module):
    def __init__(self, hp_, enc_voc, dec_voc):
        '''Attention is all you nedd. https://arxiv.org/abs/1706.03762
        Args:
            hp: Hyper Parameters
            enc_voc: vocabulary size of encoder language
            dec_voc: vacabulary size of decoder language
        '''
        super(AttModel, self).__init__()
        self.hp = hp_

        self.enc_voc = enc_voc
        self.dec_voc = dec_voc

        # encoder
        #TODO：调用基本单元模块完成编码器的词嵌入，将词索引映射为维度为hidden_units的稠密向量，并进行缩放
        self.enc_emb = embedding(self.enc_voc, self.hp.hidden_units)
        print("Embedding PASS!")
        
        #TODO：如果超参数中设置使用正弦位置编码，调用位置编码模块，维度为hidden_units，不使用零填充，也不进行缩放
        if self.hp.sinusoid:
            self.enc_positional_encoding = positional_encoding(self.hp.hidden_units, False, False)
        #TODO：否则使用可学习的嵌入方式生成位置编码，词表大小为maxlen，嵌入维度为hidden_units，不使用零填充，也不进行缩放
        else:
            self.enc_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, False, False)
        print("PositionEncoding PASS!")
        #TODO：定义dropout层
        self.enc_dropout = nn.Dropout(self.hp.dropout_rate)

        for i in range(self.hp.num_blocks):
            #TODO：调用多头注意力机制模块，隐藏维度为hidden_units，包含num_heads个注意力头，使用指定的dropout率，且不使用因果掩码
            self.__setattr__('enc_self_attention_%d' % i, multihead_attention(self.hp, self.hp.hidden_units, self.hp.num_heads, self.hp.dropout_rate, False))
            #TODO：调用前馈神经网络模块进行构建，包含一个隐藏层，中间层维度通常为输入hidden_units的4倍，输出层维度恢复到输入
            self.__setattr__('enc_feed_forward_%d' % i, feedforward(self.hp.hidden_units, [self.hp.hidden_units*4, self.hp.hidden_units]))
        
        print("LayerNormalization PASS!")
        print("MutiheadAtt PASS!")
        print("FeedForward PASS!")

        # decoder
        #TODO:调用基本单元模块完成解码器的词嵌入，调用基本单元模块将词索引映射为维度为hidden_units的稠密向量，并进行缩放
        self.dec_emb = embedding(self.dec_voc, self.hp.hidden_units)
        #TODO：如果超参数中设置使用正弦位置编码，调用位置编码模块，维度为hidden_units，不使用零填充，也不进行缩放
        if self.hp.sinusoid:
            self.dec_positional_encoding = positional_encoding(self.hp.hidden_units, False, False)
        #TODO：否则使用可学习的嵌入方式生成位置编码，词表大小为maxlen，嵌入维度为hidden_units，不使用零填充，也不进行缩放
        else:
            self.dec_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, False, False)
        #TODO：定义dropout层
        self.dec_dropout = nn.Dropout(self.hp.dropout_rate)
        for i in range(self.hp.num_blocks):
            #TODO：调用多头注意力机制模块，隐藏维度为hidden_units，包含num_heads个注意力头，使用指定的dropout率，使用掩码
            self.__setattr__('dec_self_attention_%d' % i, multihead_attention(self.hp, self.hp.hidden_units, self.hp.num_heads, self.hp.dropout_rate, True))
            #TODO：调用多头注意力机制模块，隐藏维度为hidden_units，包含num_heads个注意力头，使用指定的dropout率，不使用掩码
            self.__setattr__('dec_vanilla_attention_%d' % i, multihead_attention(self.hp, self.hp.hidden_units, self.hp.num_heads, self.hp.dropout_rate, False))
            #TODO：调用前馈神经网络模块进行构建，包含一个隐藏层，中间层维度通常为输入hidden_units的4倍，输出层维度恢复到输入
            self.__setattr__('dec_feed_forward_%d' % i, feedforward(self.hp.hidden_units, [self.hp.hidden_units*4, self.hp.hidden_units]))
        self.logits_layer = nn.Linear(self.hp.hidden_units, self.dec_voc)
        #TODO：调用标签平滑模块
        self.label_smoothing = label_smoothing()
        # self.losslayer = nn.CrossEntropyLoss(reduce=False)
        print("LabelSmoothing PASS!")
    def forward(self, x, y):
        # define decoder inputs
        # self.decoder_inputs = torch.cat([Variable(torch.ones(y[:, :1].size()).cuda() * 2).long(), y[:, :-1]], dim=-1)  # 2:<S>
        input_tensor = torch.ones(y[:, :1].size())
        if (x.is_cuda):
            input_tensor = input_tensor.cuda()
        if (x.device.type == 'mlu'):
            #TODO:将inpu_tensor放在寒武纪卡上
            input_tensor = input_tensor.to('mlu')
        self.decoder_inputs = torch.cat([Variable(input_tensor * 2).long(), y[:, :-1]], dim=-1)  # 2:<S>

        # Encoder
        #TODO：通过编码器端的词嵌入
        self.enc = self.enc_emb(x)
        # Positional Encoding
        #TODO： 使用正弦位置编码
        if self.hp.sinusoid:
            self.enc += self.enc_positional_encoding(x, self.enc)
        else:
            #self.enc += self.enc_positional_encoding(
            #    Variable(torch.unsqueeze(torch.arange(0, x.size()[1]), 0).repeat(x.size(0), 1).long().cuda()))
            enc_positional = torch.unsqueeze(torch.arange(0, x.size()[1]), 0).repeat(x.size(0), 1).long()
            if (x.is_cuda):
                enc_positional = enc_positional.cuda()
            if (x.device.type == 'mlu'):
                enc_positional = enc_positional.to('mlu')
            self.enc += self.enc_positional_encoding(Variable(enc_positional))

        #TODO：Dropout正则化                                                                                                        
        self.enc = self.enc_dropout(self.enc)
        # Blocks
        for i in range(self.hp.num_blocks):
            #实现编码器的self_attention机制
            self.enc = self.__getattr__('enc_self_attention_%d' % i)(self.enc, self.enc, self.enc)
            #实现编码器的Feed Forward机制
            self.enc = self.__getattr__('enc_feed_forward_%d' % i)(self.enc)
        # Decoder
        #TODO：解码器端的词嵌入
        self.dec = self.dec_emb(self.decoder_inputs)
        # Positional Encoding
        #TODO： 使用正弦位置编码
        if self.hp.sinusoid:
            self.dec += self.dec_positional_encoding(self.decoder_inputs, self.dec)
        else:
            #self.dec += self.dec_positional_encoding(
            #    Variable(torch.unsqueeze(torch.arange(0, self.decoder_inputs.size()[1]), 0).repeat(self.decoder_inputs.size(0), 1).long().cuda()))
            dec_positional = torch.unsqueeze(torch.arange(0, self.decoder_inputs.size()[1]), 0).repeat(self.decoder_inputs.size(0), 1).long()
            if (x.is_cuda):
                dec_positional = dec_positional.cuda()
            if (x.device.type == 'mlu'):
                dec_positional = dec_positional.to('mlu')
            #TODO：加上位置编码
            self.dec += self.dec_positional_encoding(Variable(dec_positional))




        #TODO：进行Dropout
        self.dec = self.dec_dropout(self.dec)
        # Blocks
        for i in range(self.hp.num_blocks):
            # 实现解码器内部self-attention机制
            self.dec = self.__getattr__('dec_self_attention_%d' % i)(self.dec, self.dec, self.dec)
            # 实现解码器内部vanilla attention机制
            self.dec = self.__getattr__('dec_vanilla_attention_%d' % i)(self.dec, self.enc, self.enc)
            # 实现解码器的feed forward
            self.dec = self.__getattr__('dec_feed_forward_%d' % i)(self.dec)

        # Final linear projection
        #TODO：线性映射到词表维度
        self.logits = self.logits_layer(self.dec)
        #TODO：计算概率分布，并展平成二维(batch_size * seq_len, vocab_size)
        self.probs = F.softmax(self.logits.view(-1, self.dec_voc), dim=-1)
        #TODO：获取最大概率对应的预测词索引
        _, preds_flat = torch.max(self.probs, dim=-1)
        self.preds = preds_flat.view(self.logits.size(0), self.logits.size(1))  # (B, T)
        self.istarget = (1. - y.eq(0.).float()).view(-1)
        istarget_2d = (1. - y.eq(0.).float())          # (B, T)
        self.acc = torch.sum(self.preds.eq(y).float() * istarget_2d) / torch.sum(istarget_2d)

        # Loss
        # self.y_onehot = torch.zeros(self.logits.size()[0] * self.logits.size()[1], self.dec_voc).cuda()
        self.y_onehot = torch.zeros(self.logits.size()[0] * self.logits.size()[1], self.dec_voc)
        if (x.is_cuda):
            self.y_onehot = self.y_onehot.cuda()
        if (x.device.type == 'mlu'):
            self.y_onehot = self.y_onehot.to('mlu')

        self.y_onehot = Variable(self.y_onehot.scatter_(1, y.view(-1, 1).data, 1))
        #TODO：调用标签平滑模块，得到平滑后的标签分布
        self.y_smoothed = self.label_smoothing(self.y_onehot)


        # self.loss = self.losslayer(self.probs, self.y_smoothed)
        self.loss = - torch.sum(self.y_smoothed * torch.log(self.probs), dim=-1)
        # print(self.loss)

        self.mean_loss = torch.sum(self.loss * self.istarget) / torch.sum(self.istarget)

        return self.mean_loss, self.preds, self.acc
