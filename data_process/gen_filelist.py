import os
import random
import argparse
import tqdm

def print_error(info):
    print(f"\033[31m File isn't existed: {info}\033[0m")


IndexBySinger = False
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,type=str)
    args = parser.parse_args()
    rootPath=args.input
    os.makedirs("./data/", exist_ok=True)

    all_items = []
    for spk in tqdm.tqdm(os.listdir(f"{rootPath}")):
        #print(f"{rootPath}/{spk}")
        if os.path.exists(f"{rootPath}/{spk}/CV_spk"):
            for file in os.listdir(f"{rootPath}/{spk}/CV_spk"):
                if file.endswith(".CV_spk.npy"):
                    file = file[:-11]
                    path_mel = f"{rootPath}/{spk}/mel/{file}.mel.npy"
                    path_pitch = f"{rootPath}/{spk}/pitch/{file}.pit.npy"
                    path_CV_ppg = f"{rootPath}/{spk}/CV_ppg/{file}.CV_ppg.npy"
                    path_CV_spk = f"{rootPath}/{spk}/CV_spk/{file}.CV_spk.npy"
                    path_avg_spk = f"{rootPath}/{spk}/avg_spk/avg_spk_embedding.npy"
                    has_error = 0
                    if not os.path.isfile(path_mel):
                        print_error(path_mel)
                        has_error = 1
                    if not os.path.isfile(path_avg_spk):
                        print_error(path_avg_spk)
                        has_error = 1
                    if not os.path.isfile(path_pitch):
                        print_error(path_pitch)
                        has_error = 1
                    if not os.path.isfile(path_CV_ppg):
                        print_error(path_CV_ppg)
                        has_error = 1
                    if not os.path.isfile(path_CV_spk):
                        print_error(path_CV_spk)
                        has_error = 1
                    if has_error == 0:
                        #all_items.append(f"{path_mel}|{path_avg_spk}|{path_pitch}|{path_CV_ppg}|{path_CV_spk}")
                        all_items.append(f"{rootPath}/{spk}|{file}")

    random.shuffle(all_items)
    valids = all_items
    valids.sort()
    trains = all_items
    # trains.sort()
    fw = open("data/dev.data.list", "w", encoding="utf-8")
    for strs in valids:
        print(strs, file=fw)
    fw.close()
    fw = open("data/train.data.list", "w", encoding="utf-8")
    for strs in trains:
        print(strs, file=fw)
    fw.close()
