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
import pdb

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


def pred_CV_ppg(ort_session,file, CVPPG_file):
    #print(file)
    audio, sample_rate = torchaudio.load(file)
    #print(audio.shape)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    if audio.shape[1] / 16000 > 30:
        logging.warning('do not support extract speech token for audio longer than 30s')
        speech_token = []
    else:
        feat = whisper.log_mel_spectrogram(audio, n_mels=128)
        speech_token = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),ort_session.get_inputs()[1].name: np.array([feat.shape[2]],dtype=np.int32)})[0].flatten().tolist()
    np.save(CVPPG_file, speech_token)

def gen_feats(file_list, cuda_idx):
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = [
        ("CUDAExecutionProvider", {
             "device_id": cuda_idx,
             "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
             "do_copy_in_default_stream": True,
        }),
        #"CPUExecutionProvider"
    ]
    ort_session = onnxruntime.InferenceSession("./checkpoint/speech_tokenizer_v1.onnx", sess_options=option, providers=providers)

    for file in tqdm(file_list):
        dirname, filename = os.path.split(file)
        CVPPG_dir = dirname.replace("wavs","CV_ppg")
        os.makedirs(CVPPG_dir, exist_ok=True)
        CVPPG_file = os.path.join(CVPPG_dir, filename.replace('.wav', '.CV_ppg.npy'))
        pred_CV_ppg(ort_session,file, CVPPG_file)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,required=True)
    args = parser.parse_args()
    indir=args.input
    file_list = glob.glob(indir + '/*/wavs/*.wav')
    n_gpu = torch.cuda.device_count()
    print("GPU count: ", n_gpu)
    num_processes = 3
    chunk_size = int(math.ceil(len(file_list) / num_processes))
    chunks = [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]
    print([len(c) for c in chunks])
    ctx=multiprocessing.get_context('spawn')
    pool = ctx.Pool(processes=num_processes)
    for idx,chunk in enumerate(chunks):
        pool.apply_async(gen_feats,args=(chunk,idx%n_gpu))

    pool.close()
    pool.join()
