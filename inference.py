from __future__ import absolute_import, division, print_function, unicode_literals
from fileinput import filename

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from models.meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
import librosa
import time
from models.models import Generator
from models.snakeGAN import snake_Generator, snake_Generator_v2
import parselmouth

from models.HooliGAN import HooliGenerator, HooliGenerator_snake

h = None
# device = 'cpu'
# device = 'cuda'
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    # generator = Generator(h).to(device)
    # generator = HooliGenerator(h).to(device)
    # generator = HooliGenerator_snake(h).to(device)
    # generator = snake_Generator(h).to(device)
    generator = snake_Generator_v2(h).to(device)
    # total = sum([param.nelement() for param in generator.parameters()])
    # print("Number of parameter: %.2fM" % (total/1e6))
    

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    # generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            if os.path.isdir(os.path.join(a.input_wavs_dir, filname)):
                continue
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav_ = wav
            duration = librosa.get_duration(filename=os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))
            n_frame = x.shape[2]
            # print('n_frame: ', n_frame)
            snd = parselmouth.Sound(wav_, sampling_frequency=22050)
            pitch = snd.to_pitch(time_step=snd.duration / (n_frame + 15), 
                                pitch_floor=20, 
                                pitch_ceiling=600)
            pitch = pitch.selected_array['frequency'][:n_frame]
            pitch = pitch * (pitch > 0)
            pitch = torch.FloatTensor(pitch).unsqueeze(0).to(device)
            # print('pitch: ', pitch.shape)

            beg = time.time()
            itr = 100
            for j in range(itr):
                # y_g_hat = generator(x)
                # audio = y_g_hat.squeeze()
                y_g_hat, _ = generator(x, pitch, g=None)
                # y_g_hat, _ = generator(x, pitch)
                audio = y_g_hat[0].squeeze()
            spt_time = time.time() - beg
            print('RTF: ', spt_time / itr / duration)

            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '.wav')
            # output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated_hifi.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    # config_file = './configs/HooliGAN_config.json'
    config_file = './configs/v2.json'
    # config_file = '/apdcephfs/private_sipanli/HiFi-GAN-V1-LJ/LJ_V1/config.json'
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

