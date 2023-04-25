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




        

