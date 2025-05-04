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

def pred_CV_spk(ort_session,file, CVspk_file):
    audio, sample_rate = torchaudio.load(file)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)

    feat = kaldi.fbank(audio,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten()
    np.save(CVspk_file, embedding)

def gen_feats(file_list, cuda_idx):
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = [
        #("CUDAExecutionProvider", {
        #     "device_id": cuda_idx,
        #     "arena_extend_strategy": "kNextPowerOfTwo",
        #     "cudnn_conv_algo_search": "EXHAUSTIVE",
        #     "do_copy_in_default_stream": True,
        # }),
        "CPUExecutionProvider"
    ]
    ort_session = onnxruntime.InferenceSession("checkpoint/campplus.onnx", sess_options=option, providers=providers)
    #print(ort_session.get_providers())

    for file in tqdm(file_list):
        try:
            dirname, filename = os.path.split(file)
            CVspk_dir = dirname.replace("wavs","CV_spk")
            os.makedirs(CVspk_dir, exist_ok=True)
            CVspk_file = os.path.join(CVspk_dir, filename.replace('.wav', '.CV_spk.npy'))
            pred_CV_spk(ort_session,file, CVspk_file)

        except Exception as e:
            print(f"Error occurred in subprocess: {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,required=True)
    args = parser.parse_args()
    indir=args.input
    file_list = glob.glob(indir + '/*/wavs/*.wav')
    num_processes = 5
    chunk_size = int(math.ceil(len(file_list) / num_processes))
    chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]
    print([len(c) for c in chunks])
    ctx=multiprocessing.get_context('spawn')
    pool = ctx.Pool(processes=num_processes)
    for idx,chunk in enumerate(chunks):
        pool.apply_async(gen_feats,args=(chunk,0))

    pool.close()
    pool.join()
