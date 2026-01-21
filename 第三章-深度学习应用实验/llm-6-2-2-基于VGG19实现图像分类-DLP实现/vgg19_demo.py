# -*- coding: UTF-8 -*-
import pycnnl
import time
import numpy as np
import os
import scipy.io
from PIL import Image
import imageio.v2 as imageio


def make_int_vector(*vals: int) -> "pycnnl.IntVector":
    """构造 IntVector 小工具。"""
    v = pycnnl.IntVector(len(vals))
    for i, x in enumerate(vals):
        v[i] = int(x)
    return v


class VGG19(object):
    def __init__(self):
        # set up net
        self.net = pycnnl.CnnlNet()
        self.input_quant_params = []
        self.filter_quant_params = []

    def build_model(self, param_path='../../imagenet-vgg-verydeep-19.mat'):
        """
        按 VGG19 结构搭建网络：
        conv blocks (conv+relu×n + pool) × 5 + fc6/fc7/fc8 + softmax
        """
        self.param_path = param_path

        # CNNL 这边固定按 NCHW 认为输入：N=1,C=3,H=224,W=224
        self.net.setInputShape(1, 3, 224, 224)

        in_channels = 3
        h = w = 224

        def add_conv(layer_name, out_channels,
                     kernel_size=3, stride=1, padding=1, with_relu=True):
            """
            添加一层 conv + 可选 relu，同时更新当前 feature map 形状
            """
            nonlocal in_channels, h, w

            input_shape = make_int_vector(1, in_channels, h, w)
            # 这里按照 pycnnl 文档：createConvLayer(name, input_shape, out_c, k, stride, pad, dilation)
            self.net.createConvLayer(
                layer_name,
                input_shape,
                out_channels,
                kernel_size,
                stride,
                padding,
                1  # dilation
            )

            # conv 后的空间尺寸
            h = (h - kernel_size + 2 * padding) // stride + 1
            w = (w - kernel_size + 2 * padding) // stride + 1
            in_channels = out_channels

            if with_relu:
                self.net.createReLuLayer(layer_name.replace("conv", "relu"))

        def add_pool(layer_name, pool_size=2, stride=2):
            """
            添加池化层，并更新空间尺寸
            """
            nonlocal h, w
            input_shape = make_int_vector(1, in_channels, h, w)
            # 和你原始“正确版本”一致：使用 createPoolingLayer
            self.net.createPoolingLayer(layer_name, input_shape, pool_size, stride)
            h //= stride
            w //= stride

        # VGG19 的每个 stage 通道数和 conv 层数（和你原来那版完全一致）
        nchannels = [3, 64, 128, 256, 512, 512]
        nblocks   = [0,  2,   2,   4,   4,   4]

        # 5 个 conv block
        for stage in range(1, 6):
            for block in range(1, nblocks[stage] + 1):
                add_conv(f"conv{stage}_{block}", nchannels[stage])
            add_pool(f"pool{stage}")

        def add_fc(name, in_features, out_features, with_relu=True):

            input_shape  = make_int_vector(1, 1, 1, in_features)
            weight_shape = make_int_vector(1, 1, in_features, out_features)
            output_shape = make_int_vector(1, 1, 1, out_features)
            self.net.createMlpLayer(name, input_shape, weight_shape, output_shape)
            if with_relu:
                self.net.createReLuLayer(name.replace("fc", "relu"))

        # 此时 feature map 形状：in_channels × h × w
        add_fc("fc6", in_channels * h * w, 4096)
        add_fc("fc7", 4096, 4096)
        add_fc("fc8", 4096, 1000, with_relu=False)

        # softmax：输入 shape 为 (1,1,1000)
        softmax_input = make_int_vector(1, 1, 1000)
        self.net.createSoftmaxLayer("softmax", softmax_input, 1)

    def load_model(self):

        print('Loading parameters from file ' + self.param_path)
        params = scipy.io.loadmat(self.param_path)

        # 图像均值：normalization[0][0][0] -> H×W×3，然后对 H/W 求平均
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))

        count = 0
        for idx in range(self.net.size()):
            layer_name = self.net.getLayerName(idx)

            # 卷积层
            if 'conv' in layer_name:
                # 和你原来那版一致：直接通过 idx 取 (weight, bias)
                weight, bias = params['layers'][0][idx][0][0][0][0]

                # 原来用：einops.rearrange(weight, 'h w i o -> o h w i').flatten()
                # 等价写法：np.transpose(weight, (3, 0, 1, 2)).flatten()
                weight = np.transpose(weight, (3, 0, 1, 2)).flatten().astype(float)
                bias   = bias.reshape(-1).astype(float)

                self.net.loadParams(idx, weight, bias)
                count += 1

            # 全连接层
            if 'fc' in layer_name:
                # 同样按 idx 直接取
                weight, bias = params['layers'][0][idx][0][0][0][0]


                weight = weight.reshape(
                    weight.shape[0] * weight.shape[1] * weight.shape[2],
                    weight.shape[3]
                )
                weight = weight.flatten().astype(float)
                bias   = bias.reshape(-1).astype(float)

                self.net.loadParams(idx, weight, bias)
                count += 1

        print("Loaded params for", count, "layers.")

    def load_image(self, image_dir):

        self.image = image_dir
        image_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

        print('Loading and preprocessing image from ' + image_dir)
        input_image = imageio.imread(image_dir)
        pil_img = Image.fromarray(input_image)
        pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)

        # HWC float32
        input_image = np.array(pil_img, dtype=np.float32)
        input_image -= image_mean  # 按通道减均值

        # [1, H, W, C]
        input_image = np.reshape(input_image, [1] + list(input_image.shape))

        # 你那版是直接 flatten（NHWC 展开），这里保持完全一致：
        input_data = input_image.flatten().astype(float)

        self.net.setInputData(input_data)

    def forward(self):
        return self.net.forward()
    
    def get_top5(self, label):
        start = time.time()
        self.forward()
        end = time.time()

        result = self.net.getOutputData()

        # loading labels
        labels = []
        with open('../synset_words.txt', 'r') as f:
            labels = f.readlines()

        # print results（保持原格式不动）
        top1 = False
        top5 = False
        print('------ Top 5 of ' + self.image + ' ------')
        prob = sorted(list(result), reverse=True)[:6]
        if result.index(prob[0]) == label:
            top1 = True
        for i in range(5):
            top = prob[i]
            idx = result.index(top)
            if idx == label:
                top5 = True
            print('%f - ' % top + labels[idx].strip())

        print('inference time: %f' % (end - start))
        return top1, top5
    
    def evaluate(self, file_list):
        top1_num = 0
        top5_num = 0
        total_num = 0

        start = time.time()
        with open(file_list, 'r') as f:
            file_list = f.readlines()
            total_num = len(file_list)
            for line in file_list:
                image = line.split()[0].strip()
                label = int(line.split()[1].strip())
                # 这里用 self 调用，效果和你之前用全局 vgg 一样
                self.load_image(image)
                top1, top5 = self.get_top5(label)
                if top1:
                    top1_num += 1
                if top5:
                    top5_num += 1
        end = time.time()

        print('Global accuracy : ')
        print('accuracy1: %f (%d/%d) ' %
              (float(top1_num) / float(total_num), top1_num, total_num))
        print('accuracy5: %f (%d/%d) ' %
              (float(top5_num) / float(total_num), top5_num, total_num))
        print('Total execution time: %f' % (end - start))


if __name__ == '__main__':
    vgg = VGG19()
    vgg.build_model()
    vgg.load_model()
    vgg.evaluate('../file_list')
