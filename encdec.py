import torch as t
import torch.nn as nn
from resnet import Resnet1D
from utils.torch_utils import assert_shape

class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, 
                 res_scale=False):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        for i in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(input_emb_width if i == 0 else width, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, res_scale),
            )
            blocks.append(block)
        block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
        blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class DecoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, res_scale=False, reverse_decoder_dilation=False):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
        blocks.append(block)
        for i in range(down_t):
            block = nn.Sequential(
                Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, res_scale=res_scale, reverse_dilation=reverse_decoder_dilation),
                nn.ConvTranspose1d(width, input_emb_width if i == (down_t - 1) else width, filter_t, stride_t, pad_t)
            )
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.down_t = down_t
        self.stride_t = stride_t

        block_kwargs_copy = dict(**block_kwargs)
        self.block = EncoderConvBlock(input_emb_width, output_emb_width, 
                                      down_t, stride_t, **block_kwargs_copy)

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        x = self.block(x)
        emb, T = self.output_emb_width, T // (self.stride_t ** self.down_t)
        assert_shape(x, (N, emb, T))
        return x

class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width

        self.down_t = down_t
        self.stride_t = stride_t

        self.block = DecoderConvBlock(output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs)
        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))
        x = self.block(x)
        emb, T = self.output_emb_width, T * (self.stride_t ** self.down_t)
        assert_shape(x, (N, emb, T))
        x = self.out(x)
        return x
