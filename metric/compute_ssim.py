# Copyright 2022 Tencent Inc.
# Author     : shaunxliu 
# Description: Compute SSIM measure.

import os
import glob
import argparse

import torch
import numpy as np
from tqdm import tqdm

# import ssim as SSIM
from src.utils.ssim import SSIM
from src.utils.audio import load_wav, melspectrogram
from src.utils.utils import str2bool


def compute_mel(wav_path, args):
    wav = load_wav(wav_path, sr=args.sample_rate)
    mel_spec = melspectrogram(wav, args).T

    return mel_spec


def calculate_ssim(args):
    input_files = glob.glob(f"{args.deg_dir}/*.wav")
    if len(input_files) < 1:
        raise RuntimeError(f"Found no wavs in {args.ref_dir}")

    ssim_obj = SSIM(data_range=args.max_abs_value, channel=1, size_average=False)
    ssims = []

    for deg_wav in tqdm(input_files):
        ref_wav = os.path.join(args.ref_dir, os.path.basename(deg_wav))
        ref_mel = compute_mel(ref_wav, args)
        deg_mel = compute_mel(deg_wav, args)

        ref_mel = torch.from_numpy(ref_mel).view(1, 1, ref_mel.shape[0], ref_mel.shape[1])
        deg_mel = torch.from_numpy(deg_mel).view(1, 1, deg_mel.shape[0], deg_mel.shape[1])

        min_len = min(ref_mel.shape[-2], deg_mel.shape[-2])
        ref_mel = ref_mel[:, :, :min_len, :]
        deg_mel = deg_mel[:, :, :min_len, :]

        ssim = ssim_obj(deg_mel, ref_mel)
        ssims.append(ssim[0].item())

    return np.mean(ssims)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute SSIM")

    parser.add_argument('--ref_dir', '-r', required=True)
    parser.add_argument('--deg_dir', '-d', required=True)

    mel_group = parser.add_argument_group(title="Mel options")
    mel_group.add_argument('--sample_rate', type=int, default=24000)
    mel_group.add_argument('--n_fft', type=int, default=1024)
    mel_group.add_argument('--acoustic_dim', type=int, default=80)
    mel_group.add_argument('--hop_size', type=int, default=256)
    mel_group.add_argument('--win_size', type=int, default=1024)
    mel_group.add_argument('--min_level_db', type=int, default=-100)
    mel_group.add_argument('--ref_level_db', type=int, default=20)
    # mel_group.add_argument('--symmetric_acoustic', type=str2bool, default=True)
    # mel_group.add_argument('--signal_normalization', type=str2bool, default=True)
    # mel_group.add_argument('--allow_clipping_in_normalization', type=str2bool, default=True)
    mel_group.add_argument('--symmetric_acoustic', default=True)
    mel_group.add_argument('--signal_normalization', default=True)
    mel_group.add_argument('--allow_clipping_in_normalization', default=True)
    mel_group.add_argument('--max_abs_value', type=float, default=1)
    mel_group.add_argument('--fmin', type=int, default=0)
    mel_group.add_argument('--fmax', type=int, default=12000)

    args = parser.parse_args()
    # SSIM requires data to have symmetric value range around 0
    args.symmetric_acoustic = True

    ssim_result = calculate_ssim(args)
    print(f"SSIM: {ssim_result}")
