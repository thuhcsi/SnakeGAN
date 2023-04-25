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

from models.FreGAN import ResWiseMultiPeriodDiscriminator, ResWiseMultiScaleDiscriminator

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
    total = sum([param.nelement() for param in generator.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))


    if h.stftloss:
        MRSTFT = MultiResolutionSTFTLoss(factor_mag=h.factor_mag, factor_sc=h.factor_sc).cuda()

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')

    steps = 0
    if cp_g is None:
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        generator.load_state_dict(state_dict_g['generator'])
        steps = state_dict_g['steps'] + 1
        last_epoch = -1 #state_dict_g['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank], find_unused_parameters=True).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)
    print('trainset: ', len(trainset))

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

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if epoch == 0:
            print("Number of parameter: %.2fM" % (total/1e6))
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
    
        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel, pitch, uv, = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            pitch = torch.autograd.Variable(pitch.to(device, non_blocking=True))
            uv = torch.autograd.Variable(uv.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)
            # g = torch.autograd.Variable(y.to(device, non_blocking=True))

            # y_g_hat = generator(x)
            # y_g_hat = generator(x, pitch)
            # y_g_hat, signal = generator(x, pitch, uv)
            y_g_hat, signal = generator(x, pitch, g=None, uv=None)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)


            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            # STFT loss
            if h.stftloss:
                # print('y_g_hat: ', y_g_hat.shape)
                y_g_hat = y_g_hat.squeeze(1)
                y = y.squeeze(1)
                loss_gen_mag, loss_gen_sc = MRSTFT(y_g_hat, y)
                loss_gen_all = (loss_gen_mag + loss_gen_sc)

                # signal, noise
                #print('signal: ', signal.shape, noise.shape, y_g_hat.shape, y.shape)
                
                # hooligan 
                loss_gen_mag, loss_gen_sc = MRSTFT(signal.squeeze(1), y)
                loss_gen_all += (loss_gen_mag + loss_gen_sc)


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

                            # hooligan
                            loss_gen_mag_before, loss_gen_sc = MRSTFT(signal.squeeze(1), y)
                    
                    # hooligan
                    print('Epoch : {:d}, Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, loss_gen_mag_before : {:4.3f}, loss_gen_mag_after : {:4.3f}, s/b : {:4.3f}'.
                          format(epoch, steps, loss_gen_all, mel_error, loss_gen_mag_before.item(), loss_gen_mag_after.item(), time.time() - start_b))

                    # print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, loss_gen_mag_after : {:4.3f}, s/b : {:4.3f}'.
                    #       format(steps, loss_gen_all, mel_error, loss_gen_mag_after.item(), time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict(),
                                    'steps': steps,
                                    'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    # sw.add_scalar("training/loss_gen_mag_before", loss_gen_mag_before, steps) # hooligan
                    sw.add_scalar("training/loss_gen_mag_after", loss_gen_mag_after, steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel, pitch, uv = batch
                            # y_g_hat = generator(x.to(device))\
                            # y_g_hat, signal = generator(x.to(device), pitch.to(device), uv.to(device))
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
                                              
                            if j > 200:
                                break

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='ckpts')
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
