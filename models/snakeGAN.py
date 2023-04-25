from typing import Optional

from numpy import sign

import torch
import torch.nn as nn
import torch.nn.functional as F
# LeakyRelu param?
# from snake.activations import Snake # snake actiavte

from ddsp.ddsp import DDSP, DDSP_V1, DDSP_V2, DDSP_V3
from .LVCNet import LVCNetGenerator
from .wavenet import WaveNet
from .pqmf import PQMF
from .FreGAN import DWT_1D

from BigVGAN import commons
from BigVGAN import modules

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from BigVGAN.commons import init_weights, get_padding

from torch.cuda.amp import autocast
import torchaudio.transforms as T

__all__ = ['DWT_1D']
Pad_Mode = ['constant', 'reflect', 'replicate', 'circular']


class Encoder(nn.Module):
    def __init__(self, hps):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=hps.num_mels+1, out_channels=256, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            )

    def forward(self, x):
        return self.conv(x)

'''
class HooliGenerator(nn.Module):
    def __init__(self, hps):
        super(HooliGenerator, self).__init__()
        self.encoder = Encoder(hps)
        self.ddsp = DDSP(hps.ddsp, hps.sampling_rate, hps.hop_size)
        # self.ddsp = DDSP_V1(hps.ddsp, hps.sampling_rate, hps.hop_size)

        # self.ddsp = DDSP_V2(hps.ddsp, hps.sampling_rate, hps.hop_size)
        # self.ddsp = DDSP_V3(hps.ddsp, hps.sampling_rate, hps.hop_size)
        # self.lvcnet = LVCNetGenerator()

        self.wavenet = WaveNet(hidden_dim=hps.wavenet['hidden_dim'])
        
        self.hop_size = hps.hop_size
        # snake parameter alpha, learnable
        # self.alpha = nn.Parameter(torch.tensor([5.0]), requires_grad=True)

    #     self.pqmf_layer = PQMF(N=4, taps=62, cutoff=0.15, beta=9.0)


    # def pqmf_analysis(self, x):
    #     return self.pqmf_layer.analysis(x)

    # def pqmf_synthesis(self, x):
    #     return self.pqmf_layer.synthesis(x)


    def forward(self, mel, pitch, uv=None):
        cij = torch.cat([mel, pitch.unsqueeze(1)], dim=1)
        hidden = self.encoder(cij)

        signal, harmonics, noise = self.ddsp(pitch, hidden, uv)
        # print(signal.shape, noise.shape)

        oij = torch.cat([harmonics, noise], dim=-1)
        # print(oij.shape)

        # wav = self.lvcnet(oij.permute(0,2,1), cij)
        # wav = self.lvcnet(noise.permute(0,2,1), cij)
        wav = self.wavenet(oij.permute(0,2,1), cij)
        signal = signal.permute(0,2,1)
        # print('signal: ', signal.shape, wav.shape)
        # wav = wav + signal.permute(0,2,1)

        signal = nn.functional.tanh(signal)
        wav = nn.functional.tanh(wav)
    
    # snake activation
        # signal = signal + (1 / self.alpha) * (torch.sin(self.alpha * signal) ** 2) # didn't work
        # wav = wav + (1 / self.alpha) * (torch.sin(self.alpha * wav) ** 2)
        
        # return wav, signal, noise
        return wav, signal
'''

class HooliGenerator_snake(nn.Module):
    def __init__(self, hps):
        super(HooliGenerator_snake, self).__init__()
        self.encoder = Encoder(hps)
        self.ddsp = DDSP(hps.ddsp, hps.sampling_rate, hps.hop_size)
        # self.ddsp = DDSP_V1(hps.ddsp, hps.sampling_rate, hps.hop_size)

        # self.ddsp = DDSP_V2(hps.ddsp, hps.sampling_rate, hps.hop_size)
        # self.ddsp = DDSP_V3(hps.ddsp, hps.sampling_rate, hps.hop_size)
        # self.lvcnet = LVCNetGenerator()

        self.wavenet = WaveNet(hidden_dim=hps.wavenet['hidden_dim'])
        
        self.hop_size = hps.hop_size
        # snake parameter alpha, learnable
        self.alpha = nn.Parameter(torch.tensor([5.0]), requires_grad=True)

    #     self.pqmf_layer = PQMF(N=4, taps=62, cutoff=0.15, beta=9.0)


    # def pqmf_analysis(self, x):
    #     return self.pqmf_layer.analysis(x)

    # def pqmf_synthesis(self, x):
    #     return self.pqmf_layer.synthesis(x)


    def forward(self, mel, pitch, uv=None):
        cij = torch.cat([mel, pitch.unsqueeze(1)], dim=1)
        hidden = self.encoder(cij)

        signal, harmonics, noise = self.ddsp(pitch, hidden, uv)
        # print(signal.shape, noise.shape)

        oij = torch.cat([harmonics, noise], dim=-1)
        # print(oij.shape)

        # wav = self.lvcnet(oij.permute(0,2,1), cij)
        # wav = self.lvcnet(noise.permute(0,2,1), cij)
        wav = self.wavenet(oij.permute(0,2,1), cij)
        signal = signal.permute(0,2,1)
        # print('signal: ', signal.shape, wav.shape)
        # wav = wav + signal.permute(0,2,1)

        # signal = nn.functional.tanh(signal)
        # wav = nn.functional.tanh(wav)
    
    # snake activation
        signal = signal + (1 / self.alpha) * (torch.sin(self.alpha * signal) ** 2) # didn't work
        wav = wav + (1 / self.alpha) * (torch.sin(self.alpha * wav) ** 2)
        
        # return wav, signal, noise
        return wav, signal


class AMPBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), rank=0, orig_freq=None):
        super(AMPBlock, self).__init__()

        self.upsampling_with_lfilter = T.Resample(orig_freq=orig_freq, new_freq=orig_freq*2,
                                                  resampling_method='kaiser_window',
                                                  lowpass_filter_width=12,
                                                  rolloff=0.5,
                                                  beta=4.663800127934911
                                                  )
        self.downsampling_with_lfilter = T.Resample(orig_freq=orig_freq*2, new_freq=orig_freq,
                                                  resampling_method='kaiser_window',
                                                  lowpass_filter_width=12,
                                                  rolloff=0.5,
                                                  beta=4.663800127934911
                                                  )
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs1))])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for i in range(len(self.convs2))])

    def forward(self, x, x_mask=None):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.alpha1, self.alpha2):

            with autocast(enabled=False):
                xt = self.upsampling_with_lfilter(x.float())
                xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
                xt = self.downsampling_with_lfilter(xt)

            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            
            
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
#             with autocast(enabled=False):
#                 xt = self.upsampling_with_lfilter(xt.float())
#                 xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
#                 xt = self.downsampling_with_lfilter(xt)

            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)



class snake_Generator(torch.nn.Module):
    # def __init__(self, hps, initial_channel, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0, rank=0):
    def __init__(self, hps, initial_channel=80, gin_channels=0, rank=0):    
        super(snake_Generator, self).__init__()
        self.encoder = Encoder(hps)
        self.ddsp = DDSP(hps.ddsp, hps.sampling_rate, hps.hop_size)
        self.alpha = nn.Parameter(torch.tensor([5.0]), requires_grad=True)
        # ddsp

        self.num_kernels = len(hps.resblock_kernel_sizes)
        self.num_upsamples = len(hps.upsample_rates) # 4 [8, 8, 2, 2]
        self.conv_pre = Conv1d(initial_channel, hps.upsample_initial_channel, 7, 1, padding=3)
        resblock = AMPBlock

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hps.upsample_rates, hps.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(hps.upsample_initial_channel//(2**i), hps.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        self.alphas = nn.ParameterList()

        self.alphas.append(nn.Parameter(torch.ones(1, hps.upsample_initial_channel, 1)))

        # initial_freq = [690, 5513, 11025, 22050] # how to set? 24000
        initial_freq = [750, 6000, 12000, 24000]

        for i in range(len(self.ups)):
            ch = hps.upsample_initial_channel//(2**(i+1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))

            for j, (k, d) in enumerate(zip(hps.resblock_kernel_sizes, hps.resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, rank, initial_freq[i]))


        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, hps.upsample_initial_channel, 1)

    def forward(self, x, pitch, g=None, uv=None):
        # ddsp generated speech template
        cij = torch.cat([x, pitch.unsqueeze(1)], dim=1)
        hidden = self.encoder(cij)
        signal, harmonics, _ = self.ddsp(pitch, hidden, uv)
        signal = signal.permute(0,2,1)
        signal = signal + (1 / self.alpha) * (torch.sin(self.alpha * signal) ** 2) # didn't work
        # print("signal_shape = ", signal.shape)
        
        # snake part generate the residual, condition on x (mel)
        x = self.conv_pre(x)
        # print("x_shape = ", x.shape)
        # print("g:", g)
        # if g is not None:
        #   x = x + self.cond(g)

        for i in range(self.num_upsamples): # 4 times loop
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x = self.ups[i](x)
            # print("x_up_shape = ", x.shape)
            xs = None

            for j in range(self.num_kernels):

                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)

                # print("xs_shape = ", xs.shape)

            x = xs / self.num_kernels
            # print("x_res_shape = ", x.shape)

        # ablation mv snake
        # x = x + (1 / self.alphas[i+1]) * (torch.sin(self.alphas[i+1] * x) ** 2) 
        x = self.conv_post(x)
        x = torch.tanh(x)
        x = x + signal 

        return x, signal 

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()



class snake_Generator_v2(torch.nn.Module):
    # def __init__(self, hps, initial_channel, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0, rank=0):
    def __init__(self, hps, initial_channel=80, gin_channels=0, rank=0):    
        super(snake_Generator_v2, self).__init__()
        self.encoder = Encoder(hps)
        self.ddsp = DDSP(hps.ddsp, hps.sampling_rate, hps.hop_size)
        self.alpha = nn.Parameter(torch.tensor([5.0]), requires_grad=True)
        # ddsp

        use_spectral_norm = False
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.dwt1d = DWT_1D()
        self.dwt_conv1 = norm_f(Conv1d(2, 1, 1))
        self.dwt_conv2 = norm_f(Conv1d(4, 1, 1))
        # self.dwt_conv3 = norm_f(Conv1d(8, 1, 1))
        # self.dwt_conv4 = norm_f(Conv1d(16, 1, 1))

        # DWT downsample signal generated from ddsp

        self.num_kernels = len(hps.resblock_kernel_sizes)
        self.num_upsamples = len(hps.upsample_rates) # 4 [8, 8, 2, 2]
        self.conv_pre = Conv1d(initial_channel, hps.upsample_initial_channel, 7, 1, padding=3)
        resblock = AMPBlock

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(hps.upsample_rates, hps.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(hps.upsample_initial_channel//(2**i), hps.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        self.alphas = nn.ParameterList()

        self.alphas.append(nn.Parameter(torch.ones(1, hps.upsample_initial_channel, 1)))

        # initial_freq = [690, 5513, 11025, 22050] # how to set? 24000
        initial_freq = [750, 6000, 12000, 24000]

        for i in range(len(self.ups)):
            ch = hps.upsample_initial_channel//(2**(i+1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))

            for j, (k, d) in enumerate(zip(hps.resblock_kernel_sizes, hps.resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, rank, initial_freq[i]))


        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, hps.upsample_initial_channel, 1)


    def forward(self, x, pitch, g=None, uv=None):
        # ddsp generated speech template
        cij = torch.cat([x, pitch.unsqueeze(1)], dim=1)
        hidden = self.encoder(cij)
        signal, harmonics, _ = self.ddsp(pitch, hidden, uv)
        signal = signal.permute(0,2,1)
        signal = signal + (1 / self.alpha) * (torch.sin(self.alpha * signal) ** 2) # didn't work
        # print("signal_shape = ", signal.shape)
        
        signals = []
        signals.append(signal)

        # DWT 1
        y_hi, y_lo = self.dwt1d(signal)
        y_1 = self.dwt_conv1(torch.cat([y_hi, y_lo], dim=1))
        # print("y1_shape = ", y_1.shape)
        # x_d1_high1, x_d1_low1 = self.dwt1d(y_hat)
        # y_hat_1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))
        signals.insert(0, y_1)
        
        # DWT 2
        y_d2_high1, y_d2_low1 = self.dwt1d(y_hi)
        y_d2_high2, y_d2_low2 = self.dwt1d(y_lo)
        y_2 = self.dwt_conv2(torch.cat([y_d2_high1, y_d2_low1, y_d2_high2, y_d2_low2], dim=1))
        # print("y2_shape = ", y_2.shape)
        # x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        # x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)
        # y_hat_2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))
        signals.insert(0, y_2)

        # for i in range(len(signals)):
        #     print(signals[i].shape)
        
        # snake part generate the residual, condition on x (mel)
        x = self.conv_pre(x)
        # print("x_shape = ", x.shape)
        # print("g:", g)
        # if g is not None:
        #   x = x + self.cond(g)

        for i in range(self.num_upsamples): # 4 times loop
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x = self.ups[i](x)
            # print("x_up_shape = ", x.shape)
            xs = None

            for j in range(self.num_kernels):

                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
                
                if i >= 1:
                    # print("signal[i]_shape = ", signals[i-1].shape)
                    xs += signals[i-1]
                    # xs = torch.cat([xs, signals[i-1]], dim=1)
                    
                # print("xs_shape = ", xs.shape)

            x = xs / self.num_kernels
            # print("x_res_shape = ", x.shape)

        # ablation mv snake
        x = x + (1 / self.alphas[i+1]) * (torch.sin(self.alphas[i+1] * x) ** 2)
        x = self.conv_post(x)
        x = torch.tanh(x)
        # print("x_shape = ", x.shape)
        # print("signal_shape = ", signal.shape)
        # x = x + signal 

        return x, signal 

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


# class SynthesizerTrn(nn.Module):
#   """
#   Synthesizer for Training
#   """

#   def __init__(self,
#     spec_channels,
#     segment_size,
#     resblock_kernel_sizes,
#     resblock_dilation_sizes,
#     upsample_rates,
#     upsample_initial_channel,
#     upsample_kernel_sizes,
#     gin_channels=0,
#     rank=0,

#     use_sdp=True,
#     **kwargs):

#     super().__init__()

#     self.resblock_kernel_sizes = resblock_kernel_sizes
#     self.resblock_dilation_sizes = resblock_dilation_sizes
#     self.upsample_rates = upsample_rates
#     self.upsample_initial_channel = upsample_initial_channel
#     self.upsample_kernel_sizes = upsample_kernel_sizes
#     self.segment_size = segment_size
#     self.gin_channels = gin_channels

#     self.use_sdp = use_sdp
#     self.dec = snake_Generator(spec_channels, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0, rank=rank)

#   def forward(self, x, x_lengths):

#     z_slice, ids_slice = commons.rand_slice_segments(x, x_lengths, self.segment_size)
#     o = self.dec(z_slice, g=None)

#     return o, ids_slice

#   def infer(self, x, max_len=None):

#     o = self.dec(x[:,:,:max_len], g=None)

#     return o



'''
# # DWT 3
# y_d3_high1, y_d3_low1 = self.dwt1d(y_d2_high1)
# y_d3_high2, y_d3_low2 = self.dwt1d(y_d2_low1)
# y_d3_high3, y_d3_low3 = self.dwt1d(y_d2_high2)
# y_d3_high4, y_d3_low4 = self.dwt1d(y_d2_low2)
# y_3 = self.dwt_conv3(torch.cat([y_d3_high1, y_d3_low1, y_d3_high2, y_d3_low2, y_d3_high3, y_d3_low3, y_d3_high4, y_d3_low4], dim=1))
# print("y3_shape = ", y_3.shape)
# signal_downsample.insert(0, y_3)

# # DWT 4
# y_d4_high1, y_d4_low1 = self.dwt1d(y_d3_high1)
# y_d4_high2, y_d4_low2 = self.dwt1d(y_d3_low1)
# y_d4_high3, y_d4_low3 = self.dwt1d(y_d3_high2)
# y_d4_high4, y_d4_low4 = self.dwt1d(y_d3_low2)

# y_d4_high5, y_d4_low5 = self.dwt1d(y_d3_high3)
# y_d4_high6, y_d4_low6 = self.dwt1d(y_d3_low3)
# y_d4_high7, y_d4_low7 = self.dwt1d(y_d3_high4)
# y_d4_high8, y_d4_low8 = self.dwt1d(y_d3_low4)
# y_4 = self.dwt_conv4(torch.cat([y_d4_high1, y_d4_low1, y_d4_high2, y_d4_low2, y_d4_high3, y_d4_low3, y_d4_high4, y_d4_low4, 
#     y_d4_high5, y_d4_low5, y_d4_high6, y_d4_low6, y_d4_high7, y_d4_low7, y_d4_high8, y_d4_low8] 
#     , dim=1))
# print("y4_shape = ", y_4.shape)
# signal_downsample.insert(0, y_4)
# 
'''