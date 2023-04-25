import os
from tqdm import tqdm

# with open('./all.txt', 'r') as f:
#     lines = f.readlines()
#     for line in tqdm(lines):
#         if "tx_xiao" not in line:
#             with open('./new.txt', 'a') as fi:
#                 fi.write(line)
#                 fi.close()
#         # break
tmp = []

with open('./mix_24bit.txt', 'r') as fi:
    lines = fi.readlines()
    for line in tqdm(lines):
        line = line.strip()
        line = line.split('\n')[0]
        line = line.split('.')[0]
        # print(line)
        tmp.append(line)

print(len(tmp))

with open('./all.txt', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip()
        line = line.split('|')[0]
        # print(line)
        
        if line not in tmp:
            with open('./train.txt', 'a') as file:
                file.write(line)
                file.write('|')
                file.write('\n')
                file.close()

#         # break