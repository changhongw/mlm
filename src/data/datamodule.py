import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pytorch_lightning import LightningDataModule
import h5py
import random

import torch

from src.utils.utils_mlm import (load_segs, simulate_plaintext, pad_sequence, 
                                 dali_llen, remove_duplicates, load_test_segs)

log = logging.getLogger(__name__)

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

class DALIdataModule(LightningDataModule):
    def __init__(self,
                 dali_split_dir: str, 
                 dali_feature_dir: str,
                 dali_test_dir: str,
                 batch_size: int = 32,
                 seg_lnum: int = 4, 
                 candi_ratio: float = 0.5,
                 seg_overlap: int = 0,
                 num_workers: int = 0,
                 train_ratio: float = 0.8,
                 metric_topk: int = 5):
        
        """
        Load DALI dataset and split into train, validation and test sets.

        Args:
            dali_split_dir: path to the split (csv file) of songs in the DALI dataset.
            dali_feature_dir: path to the pre-extracted features (hdf5 file) of DALI dev data; 
                features are pre-extracted to accelerate training process.
            dali_test_dir: path to the test data directory, i.e. DALI50.
            batch_size: batch size for training.
            seg_lnum: number of lines in each segment.
            candi_ratio: ratio of candidates to be kept at inference.
            seg_overlap: number of overlapping lines between segments.
            num_workers: number of workers for data loading.
            train_ratio: ratio of dev data for training.
            metric_topk: top k candidates for computing alignment and rhyme-related metrics.
        """
        
        assert dali_split_dir is not None, 'dali_split_dir is required'

        super(DALIdataModule, self).__init__()

        self.dali_split_dir = dali_split_dir
        self.dali_feature_dir = dali_feature_dir
        self.datali_test_dir = dali_test_dir

        self.seg_lnum = seg_lnum
        self.candi_ratio = candi_ratio
        self.seg_overlap = seg_overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.metric_topk = metric_topk

    def setup(self, stage: str):
        """
        Load the data and split into train, validation and test sets, including segmenting the
          full song into segments, and encode raw feature into multi-hot vectors.

        Args:
            stage:  stage of the training process, e.g. 'fit', 'validate', 'test', or 'predict'.
        """

        # load the csv file which saved randomly splitted data ids for traning, validation and testing
        df = pd.read_csv(self.dali_split_dir)
        dev_ids = df[df['split'] == 'development']['English_song_ids'].values
        test_ids = df[df['split'] == 'test']['English_song_ids'].values

        # Open the HDF5 file in read mode
        self.data = h5py.File(self.dali_feature_dir, 'r')
        all_ids = list(self.data.keys())

        dev_ids = [id for id in dev_ids if id in all_ids]
        test_ids = [id for id in test_ids if id in all_ids]

        dev_ids_rand = torch.randperm(len(dev_ids))  
        train_idx = dev_ids_rand[:int(len(dev_ids) * self.train_ratio)].tolist()
        val_idx = dev_ids_rand[int(len(dev_ids) * self.train_ratio):].tolist()
        train_ids = [dev_ids[idx] for idx in train_idx]
        val_ids = [dev_ids[idx] for idx in val_idx]

        ## check if there is any repeated songs
        assert len(set(train_ids)) == len(train_ids)
        assert len(set(val_ids)) == len(val_ids)
        assert len(set(test_ids)) == len(test_ids)

        # segmenting full song into segments
        self.splits = {
                        'train': load_segs(data=self.data, 
                                            subset_ids=train_ids,
                                            seg_lnum=self.seg_lnum,
                                            seg_overlap=self.seg_overlap),
                        'val': load_segs(data=self.data, 
                                            subset_ids=val_ids,
                                            seg_lnum=self.seg_lnum,
                                            seg_overlap=self.seg_overlap),
                        'test': load_test_segs(test_dir=self.datali_test_dir,
                                                    dali_data=self.data,
                                                    test_ids=test_ids,
                                                    seg_lnum=self.seg_lnum,
                                                    seg_overlap=0)
                        }

        # encode raw feature into multi-hot vectors
        self.splits = {key: self.splits[key] for key in self.splits.keys()}

        subset_idx = {key: remove_duplicates(self.splits[key][0]) for key in self.splits.keys()}

        for key in subset_idx.keys():
            subset_data = [self.splits[key][0][idx] for idx in subset_idx[key]]
            subset_seg_lnum = [self.splits[key][1][idx] for idx in subset_idx[key]]
            if key == 'test':
                path_truth = [self.splits[key][2][idx] for idx in subset_idx[key]] # path is at peice-level
                words_truth = [self.splits[key][3][idx] for idx in subset_idx[key]]
                self.splits[key] = [subset_data, subset_seg_lnum, path_truth, words_truth]
            else:
                self.splits[key] = [subset_data, subset_seg_lnum]

        print(len(self.splits['train'][0]), len(self.splits['val'][0]))

        self.val_batch_size = 1
        self.collate_fun = Collate_fun(dali_splits=self.splits, 
                                    dali_feature_dir=self.dali_feature_dir,
                                    seg_lnum=self.seg_lnum, 
                                    candi_ratio=self.candi_ratio,
                                    seg_overlap=self.seg_overlap,
                                    train_ratio=self.train_ratio)

        if stage == 'fit':

            self.train_ds = DALIdataset(subset_data=self.splits['train'], 
                                            subset='train')
            self.val_ds = DALIdataset(subset_data=self.splits['val'], 
                                            subset='val') 
        elif stage == 'predict': 
            self.val_ds = DALIdataset(subset_data=self.splits['val'],  
                                            subset='val') 
        else:
            self.test_ds = DALIdataset(subset_data=self.splits['test'],
                                            subset='test')

    def train_dataloader(self):

        return DataLoader(self.train_ds,
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=self.num_workers,
                        worker_init_fn=worker_init_fn,
                        persistent_workers = True,
                        pin_memory = True,
                        prefetch_factor = 2,
                        collate_fn=self.gen_collate)

    def val_dataloader(self):

        if self.train_ratio < 1.0: 
            return DataLoader(self.val_ds,
                            batch_size=self.val_batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers=self.num_workers,
                            worker_init_fn=worker_init_fn,
                            persistent_workers = True,
                            pin_memory = True,
                            prefetch_factor = 2,
                            collate_fn=self.collate_fun.collate_val)  
        else:
            dummy = TensorDataset(torch.empty(0), torch.empty(0))
            return DataLoader(dummy, batch_size=self.val_batch_size)
        
    def predict_dataloader(self):

        if self.train_ratio < 1.0: 
            return DataLoader(self.val_ds,
                            batch_size=self.val_batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers=self.num_workers,
                            worker_init_fn=worker_init_fn,
                            persistent_workers = True,
                            pin_memory = True,
                            prefetch_factor = 2,
                            collate_fn=self.collate_fun.collate_val)  
        else:
            dummy = TensorDataset(torch.empty(0), torch.empty(0))
            return DataLoader(dummy, batch_size=self.val_batch_size)
        
        
    def test_dataloader(self):
        return DataLoader(self.test_ds,
                        batch_size=1,
                        shuffle=False,
                        drop_last=False,
                        num_workers=self.num_workers,
                        collate_fn=self.collate_fun.collate_test) 
    

    # Custom collate function  
    def gen_collate(self, batch):  

        batch_music = [seq[0] for seq in batch]
        batch_lyric = [seq[1] for seq in batch]

        return pad_sequence(batch_music, batch_lyric)
    

class DALIdataset(Dataset):
        
    def __init__(self, 
                 subset_data: list, 
                 subset: str = 'train'):
        """
        Dataset for DALI dataset

        Args:
            subset_data: list of data for the subset, e.g. [note_seq, sylphone_seq, note_id].
            subset: subset of the dataset, e.g. 'train', 'val', 'test'.
        """
        
        super(DALIdataset, self).__init__()

        self.subset_data=subset_data
        self.subset = subset
        
    def __getitem__(self, idx):  

        note_id = self.subset_data[0][idx][0]
        note_seq = self.subset_data[0][idx][1]

        sylphone_seq = self.subset_data[0][idx][2]

        # original melody
        melody_ori = self.subset_data[0][idx][3].detach()

        if self.subset == 'train':
            return note_seq, sylphone_seq, note_id
        
        elif self.subset == 'test': 
            truth_path = self.subset_data[2][idx][1]
            truth_words = self.subset_data[3][idx][1]
            return note_seq, sylphone_seq, note_id, truth_path, truth_words, melody_ori
        
        else:
            return note_seq, sylphone_seq, note_id
    
    def __len__(self):
        return len(self.subset_data[0])
    

class Collate_fun():
    def __init__(self, 
                dali_splits: dict,
                dali_feature_dir: str,
                seg_lnum: int, 
                candi_ratio: int,
                seg_overlap: int,
                train_ratio: float = 1.0):
        """
        Collate function validation and test data.

        Args:
            dali_splits: dictionary containing the full training, validation, and test data. Dictionary keys include
                "train", "val", and "test"; the corresponding values are oragnized as:
                [[segment_id, note_feature, syl_feature], [segment_id, note line length, syl line length]].
            dali_feature_dir: path to "DALI_features.hdf5".
            seg_lnum: number of lines in each segment.
            candi_ratio: ratio of candidates to be kept at inference.
            seg_overlap: number of overlapping lines between segments.
            train_ratio: ratio of dev data for training.
        """

        super(Collate_fun, self).__init__()
        self.seg_lnum = seg_lnum
        self.candi_ratio = candi_ratio
        self.seg_overlap = seg_overlap
        self.dali_splits = dali_splits
        self.dali_feature_dir = dali_feature_dir

        self.dali_splits['test_shuffle'] = simulate_plaintext(dali_splits['test'])
        self.dali_len = {
                        'train': dali_llen(dali_splits=dali_splits, subset='train'),
                        'test': dali_llen(dali_splits=dali_splits, subset='test'),
                        'test_shuffle': dali_llen(dali_splits=dali_splits, subset='test_shuffle'),
                        }
        
        if train_ratio < 1.0:

            self.dali_splits['val_shuffle'] = simulate_plaintext(dali_splits['val'])
            self.dali_len.update({
                            'val': dali_llen(dali_splits=dali_splits, subset='val'),
                            'val_shuffle': dali_llen(dali_splits=dali_splits, subset='val_shuffle'),
                                })


    def collate_val(self, batch):
        subset = 'val'
        return self.collate_share(batch, subset=subset)
    
    def collate_test(self, batch):
        subset = 'test'
        return self.collate_share(batch, subset=subset)
    
    def collate_share(self, 
                      batch, 
                      subset='val') -> tuple:

        len_ids = self.dali_len[subset+'_shuffle'][0] + self.dali_len[subset][0]
        len_notes = torch.cat([self.dali_len[subset+'_shuffle'][1], self.dali_len[subset][1]], 0)
        len_syls = torch.cat([self.dali_len[subset+'_shuffle'][2], self.dali_len[subset][2]], 0)

        textbase_ids, textbase_len_notes, textbase_len_syls = len_ids, len_notes, len_syls

        if subset == 'test':
            notes, lyrics, truth_id, truth_path, truth_words, melody_ori = batch[0][0], batch[0][1], batch[0][2], batch[0][3], batch[0][4] , batch[0][5] 
        else:
            notes, lyrics, truth_id = batch[0][0], batch[0][1], batch[0][2]

        # each sample is a segment with `seg_lnum` number of lines
        note_line_len = textbase_len_notes[textbase_ids.index(truth_id)]

        len_diff = abs(torch.sum(textbase_len_syls, -1) - torch.sum(note_line_len))
        sorted_val, sorted_idx = torch.sort(len_diff, stable=True)
        candi_ids = [textbase_ids[idx] for idx in sorted_idx]
        if self.candi_ratio > 0:
            candi_ids = candi_ids[:int(len(candi_ids) * self.candi_ratio)] 

        if truth_id in candi_ids:
            truth_rank = candi_ids.index(truth_id)
            rank_val = sorted_val[truth_rank]
            truth_rank = sum(sorted_val <= rank_val).item() / len(textbase_ids)
        else:
            truth_rank = len(candi_ids) / len(textbase_ids)

        combine_data = [self.dali_splits[subset+'_shuffle'][0]+self.dali_splits[subset][0],
                    self.dali_splits[subset+'_shuffle'][1]+self.dali_splits[subset][1]]

        text_feature_dict = {item[0]: item[2] for item in combine_data[0] if item[0] in candi_ids}
        text_feature = [text_feature_dict[id] for id in candi_ids]
        
        if subset == 'val':
            
            pair = [(notes, text_feature[idx], (truth_id, candi_ids[idx]))
                    for idx in range(len(candi_ids))]
            pair.append((notes, lyrics, (truth_id, truth_id)))  
        
        else:  # subset == 'test'

            words_all = self.dali_splits[subset][3]
            words_dict = {item[0]: item[1] for item in words_all}
            candi_words = [[id, None] if '_shuffle' in id else [id, words_dict[id]] for id in candi_ids]
        
            pair = [(notes, text_feature[idx], (truth_id, candi_ids[idx]), candi_words[idx])
                    for idx in range(len(candi_ids))]
            pair.append((notes, lyrics, (truth_id, truth_id), [truth_id, truth_words]))

        batch_music = [seq[0].clone().detach() for seq in pair]
        batch_lyric = [seq[1].clone().detach() for seq in pair]

        padded_music, padded_lyric, mask_music, mask_lyric = pad_sequence(batch_music, batch_lyric)

        id_pairs = [item[2] for item in pair]

        if subset == 'test':
            words = [item[3] for item in pair]
            return padded_music, padded_lyric, mask_music, mask_lyric, id_pairs, words, (truth_rank, truth_id, truth_path, melody_ori, len(textbase_ids))
        else:
            return padded_music, padded_lyric, mask_music, mask_lyric, id_pairs, (truth_rank, truth_id, len(textbase_ids))