import os
import sys
import traceback

import librosa
import parselmouth
import matplotlib.pyplot as plt
import argparse
import logging

import numpy as np
import pyworld

sys.path.append(os.path.abspath('../'))
sys.path.append(os.getcwd())
from rmvpe.rmvpe import RMVPE
class FeatureInput(object):
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size

        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)


    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        # use 0 or 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse




if __name__ == "__main__":
    model_rmvpe = RMVPE(
        "rmvpe_pretrain/rmvpe.pt", is_half=False, device="cuda"
    )
    parser=argparse.ArgumentParser()
    parser.add_argument("-i","--input_dir",type=str,required=True)
    args=parser.parse_args()
    input_dir=args.input_dir
    f0_extract=FeatureInput()
    for root,dirs,files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                print("begin to process")
                wav, sr = librosa.load(root+"/"+file, sr=None)
                wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)

                f0 = model_rmvpe.infer_from_audio(wav16, thred=0.03)

                np.save(root+"/"+file.replace(".wav",".f0.npy"),f0)
                print("saved:",root+"/"+file.replace(".wav",".f0.npy"))
                # # 将值为1的点替换为 NaN
                # f0[f0 < 10] = np.nan
                #
                # # 绘制基频序列图
                # plt.figure(figsize=(12, 6))  # 调整图形大小
                # plt.plot(f0)
                # plt.title(f'F0 Sequence for {file.split(".")[0]}', fontsize=16)  # 调整标题字体大小
                # plt.xlabel('Time', fontsize=14)  # 调整X轴标签字体大小
                # plt.ylabel('F0', fontsize=14)  # 调整Y轴标签字体大小
                #
                # plt.grid(axis='y')  # 添加Y轴网格线
                # # 保存基频序列图
                # plot_path = os.path.join(input_dir, file.replace(".wav", ".f0_sequence.png"))
                # plt.savefig(plot_path)
                # plt.close()
                #
                # print(f"F0 sequence plot saved to {plot_path}")



