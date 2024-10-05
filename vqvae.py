import json
import numpy as np
import torch as t
import torch.nn as nn

from encdec import Encoder, Decoder, assert_shape
from vqbottleneck import NoBottleneck, Bottleneck
from utils.audio_utils import spectral_convergence, spectral_loss
from utils.audio_utils import multispectral_loss


def _loss_fn(loss_fn, x_target, x_pred, hps):
    if loss_fn == 'l1':
        return t.mean(t.abs(x_pred - x_target)) / hps.bandwidth['l1']
    elif loss_fn == 'l2':
        return t.mean((x_pred - x_target) ** 2) / hps.bandwidth['l2']
    elif loss_fn == 'linf':
        residual = ((x_pred - x_target) ** 2).reshape(x_target.shape[0], -1)
        values, _ = t.topk(residual, hps.linf_k, dim=1)
        return t.mean(values) / hps.bandwidth['l2']
    elif loss_fn == 'lmix':
        loss = 0.0
        if hps.lmix_l1:
            loss += hps.lmix_l1 * _loss_fn('l1', x_target, x_pred, hps)
        if hps.lmix_l2:
            loss += hps.lmix_l2 * _loss_fn('l2', x_target, x_pred, hps)
        if hps.lmix_linf:
            loss += hps.lmix_linf * _loss_fn('linf', x_target, x_pred, hps)
        return loss
    else:
        assert False, f"Unknown loss_fn {loss_fn}"

class VQVAE(nn.Module):
    def __init__(self, input_shape, down_t, stride_t, emb_width, l_bins, mu, 
                 commit, spectral, multispectral, multiplier=1, 
                 use_bottleneck=True, **block_kwargs):
        super().__init__()

        self.sample_length = input_shape[0]
        x_shape, x_channels = input_shape[:-1], input_shape[-1]
        self.x_shape = x_shape

        self.downsample = stride_t ** down_t  # 128

        self.z_shape = x_shape[0] // self.downsample

        self.multiplier = multiplier
        block_kwargs['width'] *= multiplier
        block_kwargs['depth'] *= multiplier

        self.encoder = Encoder(x_channels, emb_width, down_t, stride_t, block_kwargs)
        self.decoder = Decoder(x_channels, emb_width, down_t, stride_t, block_kwargs)

        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu)
        else:
            self.bottleneck = NoBottleneck()

        self.down_t = down_t
        self.stride_t = stride_t
        self.l_bins = l_bins
        self.commit = commit
        self.spectral = spectral
        self.multispectral = multispectral
        self.use_bottleneck = use_bottleneck
        self.emb_width = emb_width

    def preprocess(self, x):
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()  # (bs, T, 1) -> (bs, 1, T)
        return x

    def postprocess(self, x):
        x = x.permute(0, 2, 1)  # (bs, 1, T) -> (bs, T, 1)
        return x

    def decode(self, z):
        x_quantised = self.bottleneck.decode(z)
        x_out = self.decoder(x_quantised)
        x_out = self.postprocess(x_out)
        return x_out

    def encode(self, x):
        x_in = self.preprocess(x)
        x_out = self.encoder(x_in)
        z = self.bottleneck.encode(x_out)
        return z

    def sample(self, n_samples, device):
        if self.use_bottleneck:
            z = t.randint(0, self.l_bins, size=(n_samples, self.z_shape), device=device)
        else:
            z = t.randn(size=(n_samples, self.emb_width, self.z_shape), device=device)
        return self.decode(z)

    def forward(self, x, hps, loss_fn='l1'):

        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in)

        z, x_quantised, commit_loss, quantiser_metric = self.bottleneck(x_enc)

        x_out = self.decoder(x_quantised)
        assert_shape(x_out, x_in.shape)

        x_target = x.float()
        x_out = self.postprocess(x_out)

        recons_loss = _loss_fn(loss_fn, x_target, x_out, hps).to(x.device)

        sl = spectral_loss(x_target, x_out, hps) / hps.bandwidth['spec']
        spec_loss = t.mean(sl).to(x.device)

        msl = multispectral_loss(x_target, x_out, hps) / hps.bandwidth['spec']
        multispec_loss = t.mean(msl).to(x.device)

        loss = (recons_loss + 
                self.spectral * spec_loss + 
                self.multispectral * multispec_loss + 
                self.commit * commit_loss)

        with t.no_grad():
            sc = t.mean(spectral_convergence(x_target, x_out, hps))
            l2_loss = _loss_fn('l2', x_target, x_out, hps)
            l1_loss = _loss_fn('l1', x_target, x_out, hps)
            linf_loss = _loss_fn('linf', x_target, x_out, hps)

        metrics = dict(
            recons_loss=recons_loss,
            spectral_loss=spec_loss,
            multispectral_loss=multispec_loss,
            spectral_convergence=sc,
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            linf_loss=linf_loss,
            commit_loss=commit_loss,
            **quantiser_metric)

        for key, val in metrics.items():
            metrics[key] = val.detach()

        return x_out, loss, metrics
