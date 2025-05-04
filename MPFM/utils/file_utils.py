# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import torchaudio
import os


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lines=line.strip().split("|")
            #lists.append({"mel":lines[0],"avg_spk":lines[1],"pitch":lines[2],"speech_token":lines[3],"utt_embedding":lines[4]})
            melpath = os.path.join(lines[0],'mel',lines[1]+'.mel.npy')
            avgspkpath = os.path.join(lines[0],'avg_spk','avg_spk_embedding.npy')
            pitchpath = os.path.join(lines[0],'pitch',lines[1]+'.pit.npy')
            speechtokenpath = os.path.join(lines[0],'CV_ppg',lines[1]+'.CV_ppg.npy')
            uttembpath = os.path.join(lines[0],'CV_spk',lines[1]+'.CV_spk.npy')
            enrpath = os.path.join(lines[0],'enr',lines[1]+'.enr.npy')
            lists.append({"mel":melpath,"avg_spk":avgspkpath,"pitch":pitchpath,"speech_token":speechtokenpath,"utt_embedding":uttembpath,'enr':enrpath})
    return lists

def read_json_lists(list_file):
    lists = read_lists(list_file)
    results = {}
    for fn in lists:
        with open(fn, 'r', encoding='utf8') as fin:
            results.update(json.load(fin))
    return results

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech
