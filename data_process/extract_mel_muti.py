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
from audio import mel_spectrogram
import sys
sys.path.append('./')
def pred_CV_spk(file, mel_file,cuda_idx):
    n_fft=1024
    num_mels=100
    sampling_rate=32000
    hop_size=320
    win_size=1024
    fmin=40
    fmax=16000
    center=False
    audio, sample_rate = torchaudio.load(file)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sample_rate != 32000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=32000)(audio)
    mel=mel_spectrogram(audio.to(cuda_idx), n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=center).cpu().detach().numpy()
    np.save(mel_file, mel)

def gen_feats(file_list, cuda_idx):
    for file in tqdm(file_list):
        try:
            dirname, filename = os.path.split(file)
            mel_dir = dirname.replace("wavs","mel")
            os.makedirs(mel_dir, exist_ok=True)
            mel_file = os.path.join(mel_dir, filename.replace('.wav', '.mel.npy'))
            pred_CV_spk(file, mel_file,cuda_idx)
        except Exception as e:
            print(f"Error occurred in subprocess: {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,required=True)
    args = parser.parse_args()
    indir=args.input
    file_list = glob.glob(indir + '/*/wavs/*.wav')
    #n_gpu = torch.cuda.device_count()
    num_processes = 4
    chunk_size = int(math.ceil(len(file_list) / num_processes))
    chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]
    print([len(c) for c in chunks])
    ctx=multiprocessing.get_context('spawn')
    pool = ctx.Pool(processes=num_processes)
    for idx,chunk in enumerate(chunks):
        pool.apply_async(gen_feats,args=(chunk,0))
    pool.close()
    pool.join()
