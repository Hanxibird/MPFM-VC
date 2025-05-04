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
import librosa
sys.path.append('./')
from concurrent.futures import ThreadPoolExecutor

def pred_CV_spk(file, enr_file, cuda_idx):
    sampling_rate = 32000
    audio, sample_rate = torchaudio.load(file)

    audio = audio.to('cuda:{}'.format(cuda_idx))
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if sample_rate != 32000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=32000)(audio)


    frame_length = sampling_rate // 4 * 2
    hop_length = sampling_rate // 4
    rms = torch.sqrt(
        torch.nn.functional.pad(audio ** 2, (frame_length // 2, frame_length // 2), mode='replicate').unfold(1,frame_length,hop_length).mean(dim=-1))

    enr = rms.squeeze(0).cpu().numpy()
    np.save(enr_file, enr)

def gen_feats(file_list, cuda_idx):
    for file in tqdm(file_list):
        try:
            dirname, filename = os.path.split(file)
            mel_dir = dirname.replace("wavs","enr")
            os.makedirs(mel_dir, exist_ok=True)
            enr_file = os.path.join(mel_dir, filename.replace('.wav', '.enr.npy'))
            pred_CV_spk(file, enr_file,cuda_idx)
        except Exception as e:
            print(f"Error occurred in subprocess: {e}")


def find_wav_files(directory):
    return glob.glob(os.path.join(directory, 'wavs', '*.wav'))


def get_list(indir):
    # ???????
    subdirs = [os.path.join(indir, d) for d in os.listdir(indir) if os.path.isdir(os.path.join(indir, d))]

    wav_files = []

    # ?????????
    with ThreadPoolExecutor() as executor:
        results = executor.map(find_wav_files, subdirs)

        for result in results:
            wav_files.extend(result)
    return wav_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,required=True)
    args = parser.parse_args()
    indir=args.input
    print("begin")
    file_list = get_list(indir)

    #file_list = glob.glob(indir + '/*/wavs/*.wav')

    n_gpu = torch.cuda.device_count()
    num_processes = 32
    chunk_size = int(math.ceil(len(file_list) / num_processes))
    chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]
    print([len(c) for c in chunks])
    ctx=multiprocessing.get_context('spawn')
    pool = ctx.Pool(processes=num_processes)
    for idx,chunk in enumerate(chunks):
        pool.apply_async(gen_feats,args=(chunk,idx%n_gpu))
    pool.close()
    pool.join()
