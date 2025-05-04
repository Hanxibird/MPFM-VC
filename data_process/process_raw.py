import os
import pdb

import numpy as np
import pyloudnorm as pyln
from pydub import AudioSegment
from pydub.silence import split_on_silence
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
# 定义数据路径


# 定义响度均衡函数
def norm_wave(wave, sample_rate, target):
    if np.abs(wave).sum() > 0:
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(wave)
        wave = pyln.normalize.loudness(wave, loudness, target)
    return wave

# 定义处理音频的函数
def process_audio(file_path):
    target_dBFS = -18.0
    if 1:
    #try:
        print(file_path)
        # 加载音频文件
        audio = AudioSegment.from_wav(file_path)
        audio = audio + target_dBFS - audio.dBFS

        # 去除大块无声部分
        chunks = split_on_silence(audio, min_silence_len=1000, silence_thresh=-40)

        # 按照15s左右切片
        processed_chunks = []
        for chunk in chunks:
            if len(chunk) > 15000:  # 如果片段大于15秒
                for i in range(0, len(chunk), 15000):
                    processed_chunks.append(chunk[i:i + 15000])
            else:
                processed_chunks.append(chunk)

        # 去除小于3s的部分
        processed_chunks = [chunk for chunk in processed_chunks if len(chunk) >= 3000]

        # 响度均衡
        normalized_chunks = []
        for chunk in processed_chunks:
            samples = np.array(chunk.get_array_of_samples())
            # 将整数型数据转换为浮点型
            #samples = samples.astype(np.float32) / (2**15)
            #normalized_samples = norm_wave(samples, chunk.frame_rate, target_dBFS)
            #samples = normalized_samples
            # 将浮点型数据转换回整数型
            #normalized_samples = (normalized_samples * (2**15)).astype(np.int16)
            normalized_samples = samples
            normalized_chunk = AudioSegment(
                normalized_samples.tobytes(),
                frame_rate=chunk.frame_rate,
                sample_width=chunk.sample_width,
                channels=chunk.channels
            )
            normalized_chunks.append(normalized_chunk)

        # 下采样到32000 Hz
        resampled_chunks = [chunk.set_frame_rate(32000) for chunk in normalized_chunks]

        # 保存处理后的音频
        output_dir = os.path.join(os.path.dirname(os.path.dirname(file_path)), "wavs")
        os.makedirs(output_dir, exist_ok=True)
        for i, chunk in enumerate(resampled_chunks):
            output_path = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_chunk{i}.wav")
            chunk.export(output_path, format="wav")
    #except Exception as e:
    #    print(f"Error processing {file_path}: {e}")

parser=argparse.ArgumentParser()
parser.add_argument("-i","--input_dir",type=str,required=True)
args = parser.parse_args()


datapath = args.input_dir

# 获取所有要处理的文件路径
file_paths = []
for dirname in os.listdir(datapath):
    raw_data_path = os.path.join(datapath, dirname, "raw_data")
    if os.path.isdir(raw_data_path):
        for wav in os.listdir(raw_data_path):
            if wav.endswith(".wav"):
                file_paths.append(os.path.join(raw_data_path, wav))

for file_path in file_paths:
    process_audio(file_path)

# 使用多线程处理音频文件
#with ThreadPoolExecutor(max_workers=4) as executor:  # 你可以根据需要调整 max_workers 的数量
#    futures = [executor.submit(process_audio, file_path) for file_path in file_paths]
#    for future in as_completed(futures):
#        try:
#            future.result()
#        except Exception as e:
#            print(f"Error in future: {e}")
