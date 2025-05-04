#coding=Windows-1252
#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import torch
from tqdm import tqdm
import onnxruntime
import numpy as np
import torchaudio
import whisper
import glob
import os
import math
import multiprocessing
import torchaudio.compliance.kaldi as kaldi
import sys
sys.path.append('./')
from rmvpe.rmvpe import RMVPE

def pred_CV_spk(model_rmvpe,file, pitch_file):
    audio, sample_rate = torchaudio.load(file)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)

    pitch = model_rmvpe.infer_from_audio(audio.squeeze(0), thred=0.03)
    np.save(pitch_file, pitch)

def gen_feats(file_list, cuda_idx):
    #print(cuda_idx)
    model_rmvpe = RMVPE("checkpoint/rmvpe.pt", is_half=False, device=cuda_idx)
    #print('load done')

    for file in tqdm(file_list):
        dirname, filename = os.path.split(file)
        pitch_dir = dirname.replace("wavs","pitch")
        os.makedirs(pitch_dir, exist_ok=True)
        pitch_file = os.path.join(pitch_dir, filename.replace('.wav', '.pit.npy'))
        pred_CV_spk(model_rmvpe,file, pitch_file)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,required=True)
    args = parser.parse_args()
    indir=args.input
    file_list = glob.glob(indir + '/*/wavs/*.wav')
    n_gpu = torch.cuda.device_count()
    print("GPU count: ", n_gpu)
    num_processes = 4
    chunk_size = int(math.ceil(len(file_list) / num_processes))
    chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]
    print([len(c) for c in chunks])
    ctx=multiprocessing.get_context('spawn')
    pool = ctx.Pool(processes=num_processes)
    for idx,chunk in enumerate(chunks):
        pool.apply_async(gen_feats,args=(chunk,idx%n_gpu))

    pool.close()
    pool.join()
