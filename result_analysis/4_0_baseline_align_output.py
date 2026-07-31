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
from src.utils.utils_mlm import bresenham_line, output_alignment

subset = 'test'
data_dir = '../data/'
dali_split_dir = os.path.join(data_dir, 'DALI_English_songs.csv')
dali_feature_dir = os.path.join(data_dir, 'DALI_features.hdf5')
data_test_dir = os.path.join(data_dir, 'DALI50')

lnums = [4, 8, 12]
top_percents = [0.01]  # top 1%

def compute_baseline_alignment(seg_lnum, top_percent, method, candi_ratio):

    # path to save the alignment outputs of this method (no need for visualization, which will be finally done together for all methods)
    path_save = os.path.join(data_dir, 'test_alignment_output', f'seg{seg_lnum}_{method}_toppercent{top_percent}')
    if os.path.exists(path_save):
        shutil.rmtree(path_save)   # delete folder and all contents

    os.makedirs(path_save)         # create a new empty folder

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
    
    if method == "random":

        for batch in tqdm(batches):

            x, y, x_mask, y_mask, id_pairs, words, truth_info = batch 

            lyrics_ids = [item[1] for item in id_pairs]  # last one is to keep truth info
            truth_rank_ori, truth_id, truth_path, melody_ori, candi_num_ori = truth_info

            # ##################### randomly selected top k #####################
            select_idx = torch.randperm(len(x)-1)[:top_k]

            # identify the first non_shuffled among top_k
            rank = None
            for k in range(len(select_idx)):
                if '_shuffled' not in lyrics_ids[select_idx[k]]:
                    rank = k

                    music_mask = x_mask[0]
                    music = x[0]
                    lyrics_mask = y_mask[select_idx[rank]]
                    lyrics = y[select_idx[rank]]
                    
                    align_path = bresenham_line(sum(~music_mask).item(), sum(~lyrics_mask).item())
                    align_path = np.array(align_path)

                    seg_words = words[select_idx[rank]]

                    output_alignment(lyrics[~lyrics_mask].detach().cpu(), 
                                        align_path, melody_ori.detach().cpu(), 
                                        id_pairs[select_idx[rank]], path_save, seg_words, rank=rank)
                    
    elif method == "lengthInform":

        for batch in tqdm(batches):

            x, y, x_mask, y_mask, id_pairs, words, truth_info = batch 

            lyrics_ids = [item[1] for item in id_pairs[:-1]]  # last one is to keep truth info
            truth_rank_ori, truth_id, truth_path, melody_ori, candi_num_ori = truth_info

            # identify the first non_shuffled among top_k
            rank = None
            for k in range(top_k):
                if '_shuffled' not in lyrics_ids[k]:
                    rank = k
                
                    music = x[rank]
                    music_mask = x_mask[rank]
                    lyrics = y[rank]
                    lyrics_mask = y_mask[rank]
                    
                    align_path = bresenham_line(sum(~music_mask).item(), sum(~lyrics_mask).item())
                    align_path = np.array(align_path)

                    seg_words = words[rank]

                    output_alignment(lyrics[~lyrics_mask].detach().cpu(), 
                                        align_path, melody_ori.detach().cpu(), 
                                        id_pairs[rank], path_save, seg_words, rank=rank)
    else:
        raise NotImplementedError("Baseline method not implemented.")


def main():
    for seg_lnum in lnums:
        for top_percent in top_percents:

            compute_baseline_alignment(seg_lnum, top_percent, method="random", candi_ratio=-1)
            compute_baseline_alignment(seg_lnum, top_percent, method="lengthInform", candi_ratio=0.5)


if __name__ == "__main__":
    main()