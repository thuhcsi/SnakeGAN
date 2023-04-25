import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from models.meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models.models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from models.stftLoss import MultiResolutionSTFTLoss, STFTLoss
from common.utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

from models.HooliGAN import HooliGenerator, HooliGenerator_snake
from models.snakeGAN import snake_Generator, snake_Generator_v2

from models.FreGAN import ResWiseMultiPeriodDiscriminator, ResWiseMultiScaleDiscriminator, MultiResSpecDiscriminator

torch.backends.cudnn.benchmark = True




def train(rank, a, h):

    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    # generator = Generator(h).to(device)
    # generator = HooliGenerator(h).to(device)
    # generator = HooliGenerator_snake(h).to(device)
    # generator = snake_Generator(h).to(device)
    generator = snake_Generator_v2(h).to(device)


    # mpd = MultiPeriodDiscriminator().to(device)
    # msd = MultiScaleDiscriminator().to(device)

    mpd = ResWiseMultiPeriodDiscriminator().to(device)
    msd = ResWiseMultiScaleDiscriminator().to(device)

    if h.use_specD:
        specd = MultiResSpecDiscriminator().to(device)

    if h.stftloss:
        MRSTFT = MultiResolutionSTFTLoss(factor_mag=h.factor_mag, factor_sc=h.factor_sc).cuda()

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        if h.use_specD:
            print('state_dict_do["specd"]: ', state_dict_do['specd'].keys())
            specd.load_state_dict(state_dict_do['specd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if cp_g is not None:
        state_dict_g = load_checkpoint(cp_g, device)
        generator.load_state_dict(state_dict_g['generator'])
        steps = state_dict_g['steps'] + 1
        last_epoch = -1 #state_dict_g['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank], find_unused_parameters=True).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank], find_unused_parameters=True).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank], find_unused_parameters=True).to(device)
        if h.use_specD:
            specd = DistributedDataParallel(specd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if not h.use_specD:
        optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                    h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    else:
        optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters(), specd.parameters()),
                                    h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    if h.use_specD:
        specd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
    
        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel, pitch, uv = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            pitch = torch.autograd.Variable(pitch.to(device, non_blocking=True))
            uv = torch.autograd.Variable(uv.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            # y_g_hat = generator(x)
            #y_g_hat = generator(x, pitch)
            # y_g_hat, signal, _ = generator(x, pitch, uv) # noise didn't calculate loss
            
            # distributed
            y_g_hat, signal = generator(x, pitch, g=None, uv=None) 
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            # MSpecD
            if h.use_specD:
                y_dp_hat_r, y_dp_hat_g, _, _ = specd(y, y_g_hat.detach())
                loss_disc_p, losses_disc_p_r, losses_disc_p_g = discriminator_loss(y_dp_hat_r, y_dp_hat_g)
                loss_disc_all += loss_disc_p

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45


            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            if h.use_specD:
                y_dp_hat_r, y_dp_hat_g, fmap_p_r, fmap_p_g = specd(y, y_g_hat)
                loss_fm_p = feature_loss(fmap_p_r, fmap_p_g)
                loss_gen_p, losses_gen_p = generator_loss(y_dp_hat_g)
                loss_gen_all += (loss_gen_p + loss_fm_p)
            

            # STFT loss
            if h.stftloss:
                # print('y_g_hat: ', y_g_hat.shape)
                y_g_hat = y_g_hat.squeeze(1)
                y = y.squeeze(1)
                loss_gen_mag, loss_gen_sc = MRSTFT(y_g_hat, y)
                loss_gen_all += (loss_gen_mag + loss_gen_sc)

                # signal, noise
                #print('signal: ', signal.shape, noise.shape, y_g_hat.shape, y.shape)
                loss_gen_mag, loss_gen_sc = MRSTFT(signal.squeeze(1), y)
                loss_gen_all += (loss_gen_mag + loss_gen_sc)

                # loss_gen_mag, loss_gen_sc = MRSTFT(noise.squeeze(), y)
                # loss_gen_all += (loss_gen_mag + loss_gen_sc)


            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                        if h.stftloss:
                            # print('y_g_hat: ', y_g_hat.shape)
                            y_g_hat = y_g_hat.squeeze(1)
                            y = y.squeeze(1)
                            loss_gen_mag_after, loss_gen_sc = MRSTFT(y_g_hat, y)

                            # signal, noise
                            #print('signal: ', signal.shape, noise.shape, y_g_hat.shape, y.shape)
                            loss_gen_mag_before, loss_gen_sc = MRSTFT(signal.squeeze(), y)

                    print('Epochs : {:d}, Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, loss_gen_mag_before : {:4.3f}, loss_gen_mag_after : {:4.3f}, s/b : {:4.3f}'.
                          format(epoch, steps, loss_gen_all, mel_error, loss_gen_mag_before.item(), loss_gen_mag_after.item(), time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    d_state_dict = {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch}
                    if h.use_specD:
                        d_state_dict['specd'] = (specd.module if h.num_gpus > 1 else specd).state_dict()
                    save_checkpoint(checkpoint_path, d_state_dict)

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/loss_gen_mag_before", loss_gen_mag_before, steps)
                    sw.add_scalar("training/loss_gen_mag_after", loss_gen_mag_after, steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel, pitch, uv = batch
                            # y_g_hat = generator(x.to(device))
                            y_g_hat, signal = generator(x.to(device), pitch.to(device), uv.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                sw.add_audio('generated/y_ddsp{}'.format(j), signal[0], steps, h.sampling_rate)

                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                                y_ddsp_spec = mel_spectrogram(signal.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_ddsp_spec_{}'.format(j),
                                              plot_spectrogram(y_ddsp_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='/apdcephfs/share_1316500/shaunxliu/data/mix292spks/wav_mono_24k_16b_norm-6db')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='/apdcephfs/private_sipanli/HooliGAN/mix/new.txt')
    parser.add_argument('--input_validation_file', default='/apdcephfs/private_sipanli/HooliGAN/mix/new_val.txt')
    parser.add_argument('--checkpoint_path', default='/apdcephfs/private_sipanli/HooliGAN/outputs/snakeganv2')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
