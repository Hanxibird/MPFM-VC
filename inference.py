import pdb
import librosa

import torchaudio
import os
import sys
import torch
import torch.nn.functional as F
sys.path.append('third_party/Matcha-TTS')
from hyperpyyaml import load_hyperpyyaml
from MPFM.cli.frontend import MPFMFrontEnd
from MPFM.bin.train import load_model
from omegaconf import OmegaConf
from scipy.io.wavfile import write
from bigvgan.model.generator import Generator
from rmvpe.rmvpe import RMVPE
import argparse
import numpy as np
import math
import pickle
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def load_bigv_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    saved_state_dict = checkpoint_dict["model_g"]
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


def init_model(ckpt_path,device):
    with open('conf/MPFM.yaml', 'r') as f:
        configs = load_hyperpyyaml(f)
    time1 = time.perf_counter()
    model_flow = configs['flow']
    #model_flow.load_state_dict(torch.load(ckpt_path, map_location=device))
    saved_state_dict = torch.load(ckpt_path, map_location='cpu')
    model_flow = load_model(model_flow, saved_state_dict)
    model_flow.to(device).eval()
    time2 = time.perf_counter()
    print("FLOW模型读取：", time2 - time1)
    model_rmvpe = RMVPE(
        "checkpoint/rmvpe.pt", is_half=False, device=device
    )
    time3 = time.perf_counter()
    print("RMVPE模型读取:", time3 - time2)
    bigvgan_config = "./bigvgan/configs/nsf_bigvgan.yaml"
    bigvgan_model_path = "./checkpoint/nsf_bigvgan_pretrain_32K.pth"
    hp = OmegaConf.load(bigvgan_config)
    bigvgan_model = Generator(hp)
    load_bigv_model(bigvgan_model_path, bigvgan_model)
    bigvgan_model.eval()
    bigvgan_model.to(device)
    time4 = time.perf_counter()
    print("BIGVGAN模型读取：", time4 - time3)
    frontend = MPFMFrontEnd(configs['get_tokenizer'],
                                 configs['feat_extractor'],
                                 'checkpoint/campplus.onnx',
                                 'checkpoint/speech_tokenizer_v1.onnx',
                                 'checkpoint/spk2info.pt',
                                 False,
                                 configs['allowed_special'])
    time5 = time.perf_counter()
    print("前端模型读取", time5 - time4)
    return model_flow,model_rmvpe,bigvgan_model,frontend

def load_wav_16k_32k(wav_path):
    # 加载参考音频文件
    wav, sr = torchaudio.load(wav_path)
    wav = wav[0, :].unsqueeze(0)
    if sr != 32000:
        wav_32 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)(wav)
    else:
        wav_32 = wav
    if sr != 16000:
        wav_16 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
    else:
        wav_16 = wav
    return wav_16, wav_32
    
def get_speech_token(frontend, wav_16k):
    all_frame = wav_16k.shape[1]
    hop_frame = 2
    out_chunk = 5  # 25 S
    token_sr=16000
    hop_size=50
    speech_token_all = []
    out_index = 0
    while out_index< all_frame:
        flag=0
        if out_index == 0:  # start frame
            cut_s = 0
            cut_s_out = 0
        else:
            cut_s = out_index - hop_frame*token_sr
            cut_s_out = hop_frame*hop_size

        if out_index + (out_chunk + hop_frame)*token_sr > all_frame:  # end frame
            cut_e = all_frame
            cut_e_out = -1
            flag=1
        else:
            cut_e = out_index + (out_chunk + hop_frame)*token_sr
            cut_e_out = -1 * hop_frame*hop_size
        audio4token = wav_16k[:,cut_s:cut_e]
        speech_token, speech_token_len=frontend._extract_speech_token(audio4token)
        speech_token_all.extend(speech_token[:,cut_s_out:cut_e_out].squeeze(0))
        if flag:
            break
        out_index+=out_chunk*token_sr

    speech_token_all=torch.tensor(speech_token_all, dtype=torch.int32).unsqueeze(0)
    speech_token_all_len = speech_token_all.shape[1]
    speech_token_all_len = torch.tensor(speech_token_all_len, dtype=torch.int32)
    #for i in range(1, speech_token_all.size(1)):
    #    if speech_token_all[0,i] in [379,2443,1339,3710,2189,572,405,200,549,747,2122,1888,288,4042,347,2943,625,303,1874,638,614,494,1807,428,2945,480,724,654,3090,1053,1421,229,1760,169,410,178,474,301,627,61,278,136,142,2939,567,885,496,45,617,451,125,735,2724,657,622,91,660,111,248,521,634,586,3657,2149,80,743,689,693,330,320,438,186,651,487,1849,698,108,2,615,409,454,2230,329,616,368,3142,81,327,233,268,198] or speech_token_all[0,i] in [170, 612]:
    #        print(i)
    #        speech_token_all[0,i] = speech_token_all[0,i-1]
    return speech_token_all, speech_token_all_len

def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    #print(data1.max(),data2.max())
    #print(data1.shape, data2.shape)
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    #print(rms1.shape, rms2.shape)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (torch.pow(rms1, torch.tensor(1 - rate))* torch.pow(rms2, torch.tensor(rate - 1))).numpy()
    return data2

def split_list_by_n(list_collection, n, pre=0):
    for i in range(0, len(list_collection), n):
        yield list_collection[i-pre if i-pre>=0 else i: i + n]

def inference(model_flow,model_rmvpe,bigvgan_model,frontend, input_wav, prompt_wav, prompt_embedding,shift, rms_mix_rate, device):
    time_inf_begin=time.perf_counter()
    ######## load wav
    input_wav_16k, input_wav_32k = load_wav_16k_32k(input_wav)
    prompt_wav_16k, prompt_wav_32k = load_wav_16k_32k(prompt_wav)
    ######## prompt mel
    prompt_feat, prompt_feat_len = frontend._extract_speech_feat(prompt_wav_32k)
    print(prompt_feat.min(), prompt_feat.max())
    ######## prompt spk embedding
    if prompt_embedding is None:
        print('extract embedding from wav')
        speech_embedding = frontend._extract_spk_embedding(prompt_wav_16k)
    else:
        speech_embedding = np.load(prompt_embedding)
        speech_embedding = torch.tensor(speech_embedding, dtype=torch.float32).unsqueeze(0)
    ######## rmvpe
    input_pit = model_rmvpe.infer_from_audio(input_wav_16k.squeeze(0), thred=0.03)
    input_pit = torch.FloatTensor(input_pit)
    input_pit = input_pit * 2 ** (shift / 12)
    prompt_pit = model_rmvpe.infer_from_audio(prompt_wav_16k.squeeze(0),thred=0.03)
    prompt_pit = torch.FloatTensor(prompt_pit)
    ########### speech token
    input_token, input_token_len = get_speech_token(frontend, input_wav_16k)

    # cluster_model_path="logs/feature_and_index.pkl"
    # cluster_infer_ratio=0
    # with open(cluster_model_path, "rb") as f:
    #     cluster_model = pickle.load(f)
    # big_npy = None
    # if cluster_infer_ratio != 0:
    #     feature_index = cluster_model
    #     feat_np = np.ascontiguousarray(input_token.transpose(0, 1).cpu().numpy())
    #     if big_npy is None:
    #         big_npy = feature_index.reconstruct_n(0, feature_index.ntotal)
    #     print("starting feature retrieval...")
    #     score, ix = feature_index.search(feat_np, k=8)
    #     weight = np.square(1 / score)
    #     weight /= weight.sum(axis=1, keepdims=True)
    #     npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
    #     input_token = cluster_infer_ratio * npy + (1 - cluster_infer_ratio) * feat_np
    #     input_token = torch.tensor(input_token,dtype=torch.int32).transpose(0, 1)
    #     print("end feature retrieval...")

    np.save('data/ppg.npy',input_token)
    prompt_token, prompt_token_len = get_speech_token(frontend, prompt_wav_16k)
    ######### enr
    enr = librosa.feature.rms(y=input_wav_32k, frame_length=32000 // 100 * 2, hop_length=32000 // 100)  # 每0.25秒一个点
    input_enr = F.interpolate(torch.from_numpy(enr), size=input_token_len, mode="linear").squeeze()
    enr = librosa.feature.rms(y=prompt_wav_32k, frame_length=32000 // 100 * 2, hop_length=32000 // 100)  # 每0.25秒一个点
    prompt_enr = F.interpolate(torch.from_numpy(enr), size=prompt_token_len, mode="linear").squeeze()

    ########### cross fade args
    lg_size_token=5 # 10*20ms = 200ms cross fade
    lg_size=lg_size_token*640
    per_size=3000 # 1000*20ms = 20s per seg
    lg = torch.from_numpy(np.linspace(0, 1, lg_size) if lg_size != 0 else 0).to(device)
    input_tokens = split_list_by_n(input_token.squeeze(0), per_size, lg_size_token)

    infer_wav = torch.tensor([]).to(device)
    #############
    for k, token in enumerate(input_tokens):
        print(k)
        per_length = int(np.ceil(len(token) *32000*0.02))
        st=k*per_size-lg_size_token if k!=0 else 0
        ed=(k+1)*per_size
        pit=input_pit[st*2 :ed*2]
        enr = input_enr[st:ed]
        token_len = torch.tensor(token.shape[0], dtype=torch.int32)
        #mel = model_flow.inference_wo_prompt(token.unsqueeze(0).to(device), token_len.unsqueeze(0).to(device), speech_embedding.to(device),
        mel = model_flow.inference(token.unsqueeze(0).to(device), token_len.unsqueeze(0).to(device), speech_embedding.to(device),
                               pit.unsqueeze(0).to(device),
                               enr.unsqueeze(0).to(device), 
                               prompt_token=prompt_token.to(device),
                               prompt_token_len=prompt_token_len.unsqueeze(0).to(device),
                               prompt_feat=prompt_feat.to(device),
                               prompt_feat_len=prompt_feat_len.to(device),
                               prompt_pit=prompt_pit.unsqueeze(0).to(device),
                               prompt_enr=prompt_enr.unsqueeze(0).to(device))
        ########bigvgan
        #pit = torch.concat([pit, prompt_pit], dim=0)
        len_pit = pit.size()[0]
        len_mel = mel.size()[2]
        len_min = min(len_pit, len_mel)

        pit = pit[:len_min]
        mel = mel[:, :, :len_min]
        #np.save('mel.npy',mel.cpu().detach().numpy())
        #np.save('pit.npy',pit.cpu().detach().numpy())
        #mel = prompt_feat.transpose(1,2)[:,:,:len_min]
        with torch.no_grad():
            mel = mel.to(device)
            pit = pit.unsqueeze(0).to(device)
            _infer_wav = bigvgan_model.inference(mel, pit).squeeze(0)

        if lg_size != 0 and k != 0:
            lg1 = infer_wav[-lg_size:]
            lg2 = _infer_wav[0:lg_size]
            print(lg1.size(),lg.size(), lg2.size())
            lg_pre = lg1 * (1 - lg) + lg2 * lg
            infer_wav = torch.cat((infer_wav[0:-lg_size], lg_pre), dim=0)
            _infer_wav = _infer_wav[lg_size:]
        print(k, infer_wav.size(), _infer_wav.size())
        infer_wav = torch.cat((infer_wav, _infer_wav), dim=0)
        print(infer_wav.size())

    #audio = change_rms(input_wav_32k.squeeze(0).cpu().detach().numpy(), 32000, infer_wav.squeeze(0).cpu().detach().numpy()/32768.0, 32000, rms_mix_rate)
    audio = infer_wav.squeeze(0).cpu().detach().numpy()/32768.0

    time_inf_end=time.perf_counter()
    print("推理总时间",time_inf_end-time_inf_begin)

    return audio



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--ckpt',type=str,default='checkpoint/CosySVC_linkin/epoch_45_whole.pt')
    parser.add_argument('--ckpt',type=str,default='checkpoint/Cosy_DITflow/epoch_171_whole.pt')
    parser.add_argument('--input_wav',type=str,default='test_wav/海阔天空.wav')
    parser.add_argument('--ref_wav',type=str,default='linkin_data/chester/wavs/chester-32k_chunk44.wav')
    parser.add_argument('--ref_emb',type=str,default=None)
    parser.add_argument('--output_wav',type=str,default='test.wav')
    parser.add_argument('--shift',type=int,default=-10)
    parser.add_argument('--rms_mix_rate',type=int,default=1.0)
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    ckpt_path = args.ckpt
    input_wav=args.input_wav
    ref_wav=args.ref_wav
    ref_emb=args.ref_emb
    shift=args.shift
    rms_mix_rate = args.rms_mix_rate
    output_path=args.output_wav

    model_flow,model_rmvpe,bigvgan_model,frontend=init_model(ckpt_path,device)
    audio = inference(model_flow,model_rmvpe,bigvgan_model,frontend, input_wav, ref_wav, ref_emb, shift, rms_mix_rate, device)
    #output_path = ref_wav.split('/')[-1].split('.')[0] + '-' + input_wav.split('/')[-1].split('.')[0] + '_' + str(shift)+'-'+ckpt_path.split('/')[-3]+'.wav'

    write(output_path, 32000, audio)


