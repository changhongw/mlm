import matplotlib.pyplot as plt
plt.rcParams['figure.constrained_layout.use'] = True
import numpy as np
import os
import pandas as pd
import rootutils
import torch
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.datamodule import DALIdataModule
from src.utils.utils_mlm import bresenham_line, define_metrics, compute_stress_metrics, compute_rhyme_metrics

subset = 'test'
data_dir = '../data/'
dali_split_dir = os.path.join(data_dir, 'DALI_English_songs.csv')
dali_feature_dir = os.path.join(data_dir, 'DALI_features.hdf5')
data_test_dir = os.path.join(data_dir, 'DALI50')

lnums = [4, 8, 12]
top_k = 5
candi_ratio = 0.5

def compute_metrics_random(dataset, method, seg_lnum):

    batches = dataset.test_dataloader()

    all_metrics = define_metrics()

    if method == "lengthInform":
        
        for batch in tqdm(batches):

            x, y, x_mask, y_mask, id_pairs, words, truth_info = batch 

            lyrics_ids = [item[1] for item in id_pairs[:-1]]  # last one is to keep truth info
            truth_rank_ori, truth_id, truth_path, melody_ori, candi_num_ori = truth_info

            for k in range(top_k):
                music = x[k]
                music_mask = x_mask[k]
                lyrics = y[k]
                lyrics_mask = y_mask[k]
                
                align_path = bresenham_line(sum(~music_mask).item(), sum(~lyrics_mask).item())
                align_path = np.array(align_path)

                stress_metric = compute_stress_metrics(music, music_mask, lyrics, lyrics_mask, align_path)
                for key in stress_metric.keys():
                    all_metrics[key].append(stress_metric[key])

                lyrics_ = torch.stack([lyrics, y[-1]])
                lyrics_mask_ = torch.stack([lyrics_mask, y_mask[-1]])
                rhyme_metric = compute_rhyme_metrics(lyrics_, lyrics_mask_)
                for key in rhyme_metric.keys():
                    all_metrics[key].append(rhyme_metric[key])

    elif method == "random":

        syl_all = torch.cat([item[2] for item in dataset.splits[subset][0]], dim=0)

        for batch in tqdm(batches):

            x, y, x_mask, y_mask, id_pairs, words, truth_info = batch 

            lyrics_ids = [item[1] for item in id_pairs]  # last one is to keep truth info
            truth_rank_ori, truth_id, truth_path, melody_ori, candi_num_ori = truth_info

            # ##################### randomly selected top k #####################
            select_idx = torch.randperm(len(x)-1)[:top_k]

            for k in select_idx:
                music_mask = x_mask[0]
                music = x[0]
                lyrics_mask = y_mask[k]
                lyrics = y[k]
                
                align_path = bresenham_line(sum(~music_mask).item(), sum(~lyrics_mask).item())
                align_path = np.array(align_path)

                stress_metric = compute_stress_metrics(music, music_mask, lyrics, lyrics_mask, align_path)
                for key in stress_metric.keys():
                    all_metrics[key].append(stress_metric[key])

                lyrics_ = torch.stack([lyrics, y[-1]])
                lyrics_mask_ = torch.stack([lyrics_mask, y_mask[-1]])
                rhyme_metric = compute_rhyme_metrics(lyrics_, lyrics_mask_)
                for key in rhyme_metric.keys():
                    all_metrics[key].append(rhyme_metric[key])
        
    elif method == "reference":
    
        for batch in tqdm(batches):

            x, y, x_mask, y_mask, id_pairs, words, truth_info = batch 

            x, y, x_mask, y_mask = x[-1], y[-1], x_mask[-1], y_mask[-1]

            truth_rank_ori, truth_id, truth_path, melody_ori, candi_num_ori = truth_info

            music = x.cpu()
            lyrics = y.cpu()

            music_mask = x_mask.cpu()
            lyrics_mask = y_mask.cpu()

            truth_path = truth_path - truth_path[0]
            stress_metric = compute_stress_metrics(music, music_mask, lyrics, lyrics_mask, truth_path.numpy())
            for key in stress_metric.keys():
                all_metrics[key].append(stress_metric[key])

            rhyme_metric = compute_rhyme_metrics(y.unsqueeze(0).cpu(), y_mask.unsqueeze(0).cpu())
            for key in rhyme_metric.keys():
                all_metrics[key].append(rhyme_metric[key])

    else:
        raise ValueError("Method not implemented.")

    all_metrics_mean = {}
    all_metrics_std = {}

    for key in all_metrics.keys():
        all_metrics_mean[key] = torch.mean(torch.tensor(all_metrics[key]).float())
        all_metrics_std[key] = torch.std(torch.tensor(all_metrics[key]).float())

    # save all metrics to a csv file
    df = pd.DataFrame(all_metrics)
    df.to_csv(f'outputs/metrics_{method}_seg{seg_lnum}_top{top_k}.csv', index=False)


def main():
    for seg_lnum in lnums:

        dataset = DALIdataModule(dali_split_dir=dali_split_dir,
                                dali_feature_dir=dali_feature_dir,
                                dali_test_dir=data_test_dir,
                                batch_size=1,
                                seg_lnum=seg_lnum,
                                candi_ratio=candi_ratio,
                                seg_overlap=0,
                                train_ratio=0.8,
                                metric_topk=top_k)

        dataset.setup(stage=subset)

        compute_metrics_random(dataset, "reference", seg_lnum)
        compute_metrics_random(dataset, "lengthInform", seg_lnum)
        

    for seg_lnum in lnums:

        dataset = DALIdataModule(dali_split_dir=dali_split_dir,
                                dali_feature_dir=dali_feature_dir,
                                dali_test_dir=data_test_dir,
                                batch_size=1,
                                seg_lnum=seg_lnum,
                                candi_ratio=-1,  # use all candidates
                                seg_overlap=0,
                                train_ratio=0.8,
                                metric_topk=top_k)

        dataset.setup(stage=subset)

        compute_metrics_random(dataset, "random", seg_lnum)


if __name__ == "__main__":
    main()