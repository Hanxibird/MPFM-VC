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
sys.path.append("./")
from speakerlab.speaker import CompareSim
def pred_CV_spk(Campair_sim,file, CVspk_file):
    embedding = Campair_sim.compute_embedding(file)
    np.save(CVspk_file, np.squeeze(embedding))

def gen_feats(file_list, cuda_idx):
    Campair_sim=CompareSim("checkpoint/pretrained_eres2netv2.ckpt",cuda_idx)

    for file in tqdm(file_list):
        try:
            dirname, filename = os.path.split(file)
            CVspk_dir = dirname.replace("processed-32k","CV_spk_v2")
            os.makedirs(CVspk_dir, exist_ok=True)
            CVspk_file = os.path.join(CVspk_dir, filename.replace('.wav', '.CV_spk_v2.npy'))
            pred_CV_spk(Campair_sim,file, CVspk_file)

        except Exception as e:
            print(f"Error occurred in subprocess: {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,required=True)
    args = parser.parse_args()
    indir=args.input
    file_list = glob.glob(indir + '/*/processed-32k/*.wav')
    n_gpu = torch.cuda.device_count()
    print("GPU count: ", n_gpu)
    num_processes = 6*n_gpu
    chunk_size = int(math.ceil(len(file_list) / num_processes))
    chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]
    print([len(c) for c in chunks])
    ctx=multiprocessing.get_context('spawn')
    pool = ctx.Pool(processes=num_processes)
    for idx,chunk in enumerate(chunks):
        pool.apply_async(gen_feats,args=(chunk,idx%n_gpu))

    pool.close()
    pool.join()
