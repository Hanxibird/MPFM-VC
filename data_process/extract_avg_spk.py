import os
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', type=str, default='/data1/Cosy_dataset')
args = parser.parse_args()

input_dir = args.input_dir

for spk in tqdm(os.listdir(input_dir)):
    os.makedirs(os.path.join(input_dir, spk, "avg_spk"), exist_ok=True)

    avg_emb = None
    num = 0
    if os.path.exists(os.path.join(input_dir, spk, "CV_spk")):
        for emb in os.listdir(os.path.join(input_dir, spk, "CV_spk")):
            emb_path = os.path.join(input_dir, spk, "CV_spk", emb)
            emb_data = np.load(emb_path)

            if avg_emb is None:
                avg_emb = emb_data
            else:
                avg_emb += emb_data

            num += 1

        if num > 0:
            avg_emb = avg_emb / num
            np.save(os.path.join(input_dir, spk, "avg_spk", "avg_spk_embedding.npy"), avg_emb)
        else:
            print(f"No embeddings found for speaker {spk}")
    else:
        print(f"exit none {os.path.join(input_dir, spk, 'CV_spk')}")