import shutil

import matplotlib.pyplot as plt
plt.rcParams['figure.constrained_layout.use'] = True
import numpy as np
import os
import rootutils
import torch

from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.datamodule import DALIdataModule

subset = 'test'
data_dir = '../data/'
dali_split_dir = os.path.join(data_dir, 'DALI_English_songs.csv')
dali_feature_dir = os.path.join(data_dir, 'DALI_features.hdf5')
data_test_dir = os.path.join(data_dir, 'DALI50')

lnums = [4, 8, 12]
top_percents = [0.01]  # top 1%


def compute_baseline_alignment(seg_lnum, top_percent, method, candi_ratio):

    dataset = DALIdataModule(dali_split_dir=dali_split_dir,
                            dali_feature_dir=dali_feature_dir,
                            dali_test_dir=data_test_dir,
                            batch_size=1,
                            seg_lnum=seg_lnum,
                            candi_ratio=candi_ratio,
                            seg_overlap=0,
                            train_ratio=0.8,
                            metric_topk=5)
        
    dataset.setup(stage=subset)
    batches = dataset.test_dataloader()

    top_k = round(len(batches) * 2 * top_percent)
    shuffle_ratio_all = []
    
    if method == "random":

        for batch in tqdm(batches):

            x, y, x_mask, y_mask, id_pairs, words, truth_info = batch 

            lyrics_ids = [item[1] for item in id_pairs]  # last one is to keep truth info

            # ##################### randomly selected top k #####################
            select_idx = torch.randperm(len(x)-1)[:top_k]

            shuffle_ratio = sum([1 for k in range(len(select_idx)) if '_shuffled' in lyrics_ids[select_idx[k]]]) / top_k
            shuffle_ratio_all.append(shuffle_ratio)

    elif method == "lengthInform":

        for batch in tqdm(batches):

            x, y, x_mask, y_mask, id_pairs, words, truth_info = batch 

            lyrics_ids = [item[1] for item in id_pairs[:-1]]  # last one is to keep truth info

            shuffle_ratio = sum([1 for k in range(top_k) if '_shuffled' in lyrics_ids[k]]) / top_k
            shuffle_ratio_all.append(shuffle_ratio)
                
    else:
        raise NotImplementedError("Baseline method not implemented.")
    
    print(f"Seg lnum: {seg_lnum}, total num: {len(batches)}, top_k: {top_k}")
    print(f"Mean perturned ratio among top_percent {top_percent}: {np.mean(np.array(shuffle_ratio_all))}")

def main():
    for seg_lnum in lnums:
        for top_percent in top_percents:

            print(f"---------------- method: random ------------------")
            compute_baseline_alignment(seg_lnum, top_percent, method="random", candi_ratio=-1)
            print(f"---------------- method: lengthInform ------------------")
            compute_baseline_alignment(seg_lnum, top_percent, method="lengthInform", candi_ratio=0.5)

if __name__ == "__main__":
    main()