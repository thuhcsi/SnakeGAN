import os
from tqdm import tqdm

for file in tqdm(os.listdir('/apdcephfs/share_1316500/shaunxliu/data/mix292spks/wav_mono_24k_16b_norm-6db/')):
    # print(file)
    idx = file.split('.')[0]
    # print(idx)
    # break
    with open('./mix_train.txt', 'a') as f:
        f.write(idx)
        f.write('|')
        f.write('\n')
        f.close()
    # break