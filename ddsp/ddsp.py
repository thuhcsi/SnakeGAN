from email import header
import torch
import torch.nn as nn
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve
from .core import resample
import math
from models.wavenet import ConvInUpsampleNetwork

class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


class DDSP(nn.Module):
    def __init__(self, hps, sampling_rate, block_size):
        super().__init__()
        hidden_size = hps['hidden_size']
        n_harmonic = hps['n_harmonic']
        n_bands = hps['n_bands']

        self.use_uv = hps['use_uv']

        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3), mlp(hidden_size, hidden_size, 3)])
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size * 2 + 1, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness, uv=None):
        pitch = pitch.unsqueeze(-1)
        loudness = loudness.permute(0, 2, 1)
        # print('loudness: ', loudness.shape)
        # print('pitch: ', pitch.shape)
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        # print('hidden: ', hidden.shape)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        # print('amplitudes: ', amplitudes.shape)
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp
        # print('block_size:', self.block_size) # tensor(256, device='cuda:0')
        amplitudes = upsample(amplitudes, int(self.block_size))
        pitch = upsample(pitch, int(self.block_size))

        signal, harmonics = harmonic_synth(pitch, amplitudes, self.sampling_rate)
        if self.use_uv:
            # print('uv: ', uv.shape)
            uv = uv.unsqueeze(-1)
            uv = upsample(uv, int(self.block_size))
            # print('uv: ', uv.shape, signal.shape, harmonics.shape)
            signal = signal * uv
            harmonics = harmonics * uv

        # noise part
        # param = scale_function(self.proj_matrices[1](hidden) - 5)
        param = self.proj_matrices[1](hidden)
        # print('param: ', param.shape)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = signal + noise

        # signal = nn.functional.tanh(signal)


        return signal, harmonics, noise


class DDSP_V1(nn.Module):
    def __init__(self, hps, sampling_rate, block_size):
        super().__init__()
        hidden_size = hps['hidden_size']
        n_harmonic = hps['n_harmonic']
        n_bands = hps['n_bands']

        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        
        self.pitchUP = ConvInUpsampleNetwork(cin_channels=1)
        self.hiddenUP = ConvInUpsampleNetwork(cin_channels=hidden_size)

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 1), mlp(hidden_size, hidden_size, 1)])
        # self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size + 1, hidden_size, 1)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness):
        pitch = pitch.unsqueeze(1)

        pitch = self.pitchUP(pitch)
        loudness = self.hiddenUP(loudness)
        # print('loudness: ', loudness.shape)

        loudness = loudness.permute(0, 2, 1)
        pitch = pitch.permute(0, 2, 1)
        # print('loudness: ', loudness.shape)
        # print('pitch: ', pitch.shape)
        hidden = torch.cat([
            pitch,
            self.in_mlps[1](loudness),
        ], -1)
        # hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        # print('hidden: ', hidden.shape)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        # print('amplitudes: ', amplitudes.shape)
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        # amplitudes = upsample(amplitudes, self.block_size)
        # pitch = upsample(pitch, self.block_size)

        signal, harmonics = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        # param = scale_function(self.proj_matrices[1](hidden) - 5)
        param = self.proj_matrices[1](hidden)
        # print('param: ', param.shape)

        impulse = amp_to_impulse_response(param, 1)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            1,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        # print('signal: ', signal.shape, noise.shape)

        signal = signal + noise

        # signal = nn.functional.tanh(signal)


        return signal, harmonics, noise

    def realtime_forward(self, pitch, loudness):
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)

        gru_out, cache = self.gru(hidden, self.cache_gru)
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.sampling_rate, 1)

        omega = omega + self.phase
        self.phase.copy_(omega[0, -1, 0] % (2 * math.pi))

        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)

        harmonic = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        return signal


class DDSP_V2(nn.Module):
    def __init__(self, hps, sampling_rate, block_size):
        super().__init__()
        hidden_size = hps['hidden_size']
        n_harmonic = hps['n_harmonic']
        n_bands = hps['n_bands']

        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.pitchUP = ConvInUpsampleNetwork(cin_channels=1)
        self.hiddenUP = ConvInUpsampleNetwork(cin_channels=n_harmonic)

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 3), mlp(hidden_size, hidden_size, 3)])
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size * 2 + 1, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness):
        pitch = pitch.unsqueeze(-1)
        loudness = loudness.permute(0, 2, 1)
        # print('loudness: ', loudness.shape)
        # print('pitch: ', pitch.shape)
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        # print('hidden: ', hidden.shape)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        # print('amplitudes: ', amplitudes.shape)
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        # amplitudes = upsample(amplitudes, self.block_size)
        # pitch = upsample(pitch, self.block_size)

        #pitch = pitch.permute(0, 2, 1)
        pitch = self.pitchUP(pitch.permute(0, 2, 1)).permute(0, 2, 1)
        amplitudes = self.hiddenUP(amplitudes.permute(0, 2, 1)).permute(0, 2, 1)

        signal, harmonics = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        # param = scale_function(self.proj_matrices[1](hidden) - 5)
        param = self.proj_matrices[1](hidden)
        # print('param: ', param.shape)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = signal + noise

        # signal = nn.functional.tanh(signal)


        return signal, harmonics, noise


class DDSP_V3(nn.Module):
    def __init__(self, hps, sampling_rate, block_size):
        super().__init__()
        hidden_size = hps['hidden_size']
        n_harmonic = hps['n_harmonic']
        n_bands = hps['n_bands']

        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        self.pitchUP = ConvInUpsampleNetwork(cin_channels=1)
        self.hiddenUP = ConvInUpsampleNetwork(cin_channels=n_harmonic)

        self.in_mlps = nn.ModuleList([mlp(1, hidden_size, 1), mlp(hidden_size, hidden_size, 1)])
        self.gru = gru(2, hidden_size)
        self.out_mlp = mlp(hidden_size * 2 + 1, hidden_size, 3)

        self.proj_matrices = nn.ModuleList([
            nn.Linear(hidden_size, n_harmonic + 1),
            nn.Linear(hidden_size, n_bands),
        ])

        self.reverb = Reverb(sampling_rate, sampling_rate)

        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

    def forward(self, pitch, loudness):
        pitch = pitch.unsqueeze(-1)
        loudness = loudness.permute(0, 2, 1)
        # print('loudness: ', loudness.shape)
        # print('pitch: ', pitch.shape)
        hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)
        hidden = torch.cat([self.gru(hidden)[0], pitch, loudness], -1)
        # print('hidden: ', hidden.shape)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        # print('amplitudes: ', amplitudes.shape)
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        # amplitudes = upsample(amplitudes, self.block_size)
        # pitch = upsample(pitch, self.block_size)

        #pitch = pitch.permute(0, 2, 1)
        pitch = self.pitchUP(pitch.permute(0, 2, 1)).permute(0, 2, 1)
        amplitudes = self.hiddenUP(amplitudes.permute(0, 2, 1)).permute(0, 2, 1)

        signal, harmonics = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        # param = scale_function(self.proj_matrices[1](hidden) - 5)
        param = self.proj_matrices[1](hidden)
        # print('param: ', param.shape)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = signal + noise

        # signal = nn.functional.tanh(signal)


        return signal, harmonics, noise

if __name__ == '__main__':
    ddsp = DDSP(hidden_size=256, n_harmonic=150, n_bands=64, sampling_rate=16000,
                 block_size=160)

    pitch = torch.rand((2,100,1))
    lound = torch.rand((2,100,1))

    signal = ddsp(pitch, lound)
    print(signal.shape)

    