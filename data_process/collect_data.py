import os
import shutil
strs=["PopBuTFy"]

datapath="/data1/Sing_rawdata/ft_local"
cosypath="/data1/Cosy_dataset"
for dir_name in os.listdir(datapath):
    # 检查目录名是否包含在 strs 列表中的任何一个字符串
    if any(substring in dir_name for substring in strs):
        print(dir_name)
        os.makedirs(os.path.join(cosypath,dir_name),exist_ok=True)
        os.makedirs(os.path.join(cosypath,dir_name,"raw_data"),exist_ok=True)
        for wav in os.listdir(os.path.join(datapath,dir_name)):
            if wav.endswith(".wav"):
                shutil.copy(os.path.join(datapath,dir_name,wav),os.path.join(cosypath,dir_name,"raw_data","PopBuTFy"+"_"+wav))