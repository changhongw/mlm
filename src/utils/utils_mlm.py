from collections import Counter
import DALI as dali_code
import librosa
import matplotlib.pyplot as plt
plt.rcParams['figure.constrained_layout.use'] = True
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any

import torch
import torch.nn.functional as F

from src.utils.preprocess_utils import (clean_and_normalize_text, sylphone_encode, note_encode, 
                                        get_phone_dict, sylphone_feature_idx)

# pre-extracted set of notes and sylphones based on filtering on the development set
unique_set = np.load("./outputs/unique_set_note_sylphone.npz")
note_unique_set = unique_set["note_unique"]
syl_unique_set = unique_set["syl_unique"]

note_unique = note_unique_set.tolist()
syl_unique = syl_unique_set.tolist()

note_unique_set = set(map(tuple, note_unique_set))
syl_unique_set = set(map(tuple, syl_unique_set))

def load_segs(data: dict, 
              subset_ids: list, 
              seg_lnum: int,
              seg_overlap: int) -> tuple[list, list]:
    """
    Split full length pieces into segments with specific number of lines.

    Args:
        data: loaded HDF5 data, organized as
            "<song_id>":  
                "note_encode": torch.tensor
                "sylphone_encode": torch.tensor
                "sylphones": list, each element is a sylphone (i.e., syllable vector).
        subset_ids: song ids in the subset ("train" or "validation").
        seg_lnum: number of lines in each segment.
        seg_overlap: number of overlapping lines between segments.

    Returns:
        seg_lines: list of segments, each containing [seg_id, note_features, sylphone features].
        seg_linelen: list of segment lengths, each containing 
            [seg_id, number of notes per line, number of sylphones per line].
    """
    
    seg_lines = []
    seg_linelen = []
    for song_id in subset_ids:
        note_seq = torch.from_numpy(data[song_id]['note_encode'][:]) 
        syl_seq = torch.from_numpy(data[song_id]['sylphone_encode'][:])

        if sum(note_seq[:,-1] == 1) >= seg_lnum: # at least one segment
            note_line_idx = torch.cat((torch.tensor([-1]), torch.nonzero(note_seq[:,-1] == 1).squeeze(-1)))
            syl_line_idx = torch.cat((torch.tensor([-1]), torch.nonzero(syl_seq[:,-1] == 1).squeeze(-1)))

            assert len(note_line_idx) == len(syl_line_idx)

            note_lines = [note_seq[note_line_idx[k]+1:note_line_idx[k+1]+1] for k in range(len(note_line_idx)-1)]
            syl_lines = [syl_seq[syl_line_idx[k]+1:syl_line_idx[k+1]+1] for k in range(len(syl_line_idx)-1)]

            # filter out segments with rare line length
            lines = [[note_lines[k], syl_lines[k]] for k in range(len(note_lines))
                                                            if 
                                                            note_lines[k].size(0) >=3 and note_lines[k].size(0) <=11 and
                                                            syl_lines[k].size(0)  >=2 and syl_lines[k].size(0)  <=10]


            note_lines = [item[0] for item in lines]
            syl_lines = [item[1] for item in lines]

            note_segs = [note_lines[k:k+seg_lnum] for k in range(len(note_lines)-seg_lnum+1)[::seg_lnum-seg_overlap]] # range for start, include last
            syl_segs = [syl_lines[k:k+seg_lnum] for k in range(len(syl_lines)-seg_lnum+1)[::seg_lnum-seg_overlap]]

            note_segs_llen = [torch.tensor([line.size(0) for line in item]) for item in note_segs]
            syl_segs_llen = [torch.tensor([line.size(0) for line in item]) for item in syl_segs]

            idx = [k for k in range(len(note_segs_llen)) if note_segs_llen[k].sum() <= 1024 and syl_segs_llen[k].sum() <= 1024]  
                
            note_segs = [note_segs[k] for k in idx]
            syl_segs = [syl_segs[k] for k in idx]
            note_segs_llen = [note_segs_llen[k] for k in idx]
            syl_segs_llen = [syl_segs_llen[k] for k in idx]

            note_segs = [[note_encode(torch.cat(item, 0)).float(), torch.cat(item, 0).float()] for item in note_segs]
            syl_segs = [torch.cat(item, 0) for item in syl_segs]
            syl_segs = [torch.from_numpy(np.delete(item.numpy(), -2, axis=-1)).float() for item in syl_segs]  # remove stopwords and word index

            # filter out segments with rare notes and rare syllable vectors
            filtered_idx = [idx for idx in range(len(note_segs)) if 
                        set(map(tuple, note_segs[idx][0][:,:-1].numpy())).issubset(note_unique_set) and 
                        set(map(tuple, syl_segs[idx][:,:-1].numpy())).issubset(syl_unique_set)]
            
            note_segs = [note_segs[idx] for idx in filtered_idx]
            syl_segs = [syl_segs[idx] for idx in filtered_idx]

            note_segs_llen = [note_segs_llen[k] for k in filtered_idx]
            syl_segs_llen = [syl_segs_llen[k] for k in filtered_idx]

            seg_lines.extend([[song_id+'_'+str(k), note_segs[k][0], syl_segs[k], note_segs[k][1]] for k in range(len(note_segs))])
            seg_linelen.extend([[song_id+'_'+str(k), note_segs_llen[k], syl_segs_llen[k]] for k in range(len(note_segs_llen))])

    # shuffle order
    shuf_lines = torch.randperm(len(seg_lines))
    seg_lines = [seg_lines[idx] for idx in shuf_lines]
    seg_linelen = [seg_linelen[idx] for idx in shuf_lines]

    return seg_lines, seg_linelen

def remove_duplicates(subset_data: list) -> list:
    """
    Remove duplicate songs in the subset using hashing.

    Args:
        subset_data: list of segments, each element containing information
            [segment_id, note feature, sylphone feature].

    Return:
        subset_idx: list of segment index kepted after filtering out repeated
            segments.
    """

    seen_songs_music, seen_songs_lyric = {}, {}
    rep_song = set()
    for idx in range(len(subset_data)):
        song_music = subset_data[idx][1][:, :-1].numpy()  # pitch, (start, end, beat, downbeat, line_mark)
        song_lyrics = subset_data[idx][2][:, :66].numpy() # only sylphone
        # Create a hash from the shape and content
        song_hash_music = (song_music[:, :129].shape, song_music[:, :129].tobytes())
        song_hash_lyric = (song_lyrics.shape, song_lyrics.tobytes())

        if song_hash_music in seen_songs_music or song_hash_lyric in seen_songs_lyric or \
            len(np.unique(song_music[:, :129], axis=0)) <= 1 or len(np.unique(song_music[:, 129:129+24], axis=0)) <= 1 \
            or len(np.unique(song_music[:, 129+24:129+24*2], axis=0)) <= 1:  # ensure pitch diversity > 1/4 octave
            rep_song.add(idx)
        else:
            seen_songs_music[song_hash_music] = idx
            seen_songs_lyric[song_hash_lyric] = idx

    subset_idx = [idx for idx in range(len(subset_data)) if idx not in rep_song] # 5472

    return subset_idx

def withinsample_negatives(x: torch.tensor, 
                        y: torch.tensor, 
                        x_mask: torch.tensor,
                        y_mask: torch.tensor,
                        cross_ratio: float,
                        shuffle_ratio: float,
                        neg_num: int,
                        ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Create negative pairs by shuffling melody notes or lyrics syllable vectors within the sequence.

    Args:
        x (torch.tensor): input note features.
        y (torch.tensor): input sylphone features. 
        x_mask (torch.tensor): masks for notes.
        y_mask (torch.tensor): masks for sylphones.
        cross_ratio (float): ratio of cross-samples negatives.
        shuffle_ratio (float): ratio of shuffling within the sample.
        neg_num (int): number of negative samples.
    """

    device = x.device
    b = x.size(0)
    
    x_ori = x.clone()
    y_ori = y.clone()

    xmask_ori = x_mask.clone()
    ymask_ori = y_mask.clone()
        
    withinNeg_num = neg_num - int(neg_num * cross_ratio)

    if withinNeg_num == 0:
        return x, y, x_mask, y_mask
    else:
        ########## within-line negatives: shuffle within segment ##########
        x = torch.repeat_interleave(x, withinNeg_num, dim=0)
        y = torch.repeat_interleave(y, withinNeg_num, dim=0)
        x_mask = torch.repeat_interleave(x_mask, withinNeg_num, dim=0)
        y_mask = torch.repeat_interleave(y_mask, withinNeg_num, dim=0)

        # block-wise half shuffle
        for idx in range(len(y)):
            # get a starting point between start and middle point
            start = torch.randperm(len(y[idx, :sum(~y_mask[idx])]) - int(len(y[idx, :sum(~y_mask[idx])]) * shuffle_ratio) + 1, device=device)[0]
            shuffle_pos = torch.randperm(int(len(y[idx, :sum(~y_mask[idx])]) * shuffle_ratio), device=device) + start 
            y[idx, start:max(shuffle_pos)+1, :-1] = y[idx][shuffle_pos, :-1].clone()

        for idx in range(len(x)):
            start = torch.randperm(len(x[idx, :sum(~x_mask[idx])]) - int(len(x[idx, :sum(~x_mask[idx])]) * shuffle_ratio) + 1, device=device)[0]
            shuffle_pos = torch.randperm(int(len(x[idx, :sum(~x_mask[idx])]) * shuffle_ratio), device=device) + start 
            x[idx, start:max(shuffle_pos)+1, :-1] = x[idx][shuffle_pos, :-1].clone()

        x = x.reshape(b, -1, x.size(1), x.size(2))
        x = torch.cat([x_ori.unsqueeze(1), x], 1)

        y = y.reshape(b, -1, y.size(1), y.size(2))
        y = torch.cat([y_ori.unsqueeze(1), y], 1)

        x_mask = x_mask.reshape(b, -1, x_mask.size(1))
        x_mask = torch.cat([xmask_ori.unsqueeze(1), x_mask], 1)

        y_mask = y_mask.reshape(b, -1, y_mask.size(1))
        y_mask = torch.cat([ymask_ori.unsqueeze(1), y_mask], 1)

        x = torch.flatten(x, start_dim=0, end_dim=1)
        y = torch.flatten(y, start_dim=0, end_dim=1)

        x_mask = torch.flatten(x_mask, start_dim=0, end_dim=1)
        y_mask = torch.flatten(y_mask, start_dim=0, end_dim=1)

        return x, y, x_mask, y_mask


def simulate_plaintext(data: tuple) -> tuple[list, list]:
    """
    Simulate the plaintext by random sampling from the input data.

    Args: 
        data (tuple):
            data[0] (list): each element is a segment, with information [seg_id, note feature, sylphone feature]. 
            data[1] (list): each element is a segment, with information [seg_id, note line length, sylphone line length]. 
            data[2] (list): each element is the reference alignment path for the segment: [seg_id, path matrix]
            data[3] (list): each element contains the word information of the segment: [seg_id, [words, word index]].

    Returns:
        shuffled_data (list): segments with sylphones randomly sampled.
        shuffle_len (list): segment length information.
    """

    # Shuffle the lines of the input data
    shuffled_data = []
    shuffle_len = []
    syl_all = torch.cat([item[2] for item in data[0]], dim=0)
    for k in range(len(data[0])):
        # shuffle all syllable vectors and then sample
        seg = data[0][k]
        new_seg = syl_all[torch.randperm(syl_all.size(0))[:seg[2].size(0)],:]
        new_seg[:,-1] = seg[2][:,-1] # keep the line marker
        shuffled_data.append([seg[0]+'_shuffled', seg[1], new_seg])
        # update length
        seg_len = data[1][k]
        shuffle_len.append([seg_len[0]+'_shuffled', seg_len[1], seg_len[2]])

    return [shuffled_data, shuffle_len]

def dali_llen(dali_splits: dict, 
              subset: str='train') -> tuple[torch.tensor, torch.tensor]:
    """
    Collect segment length information for quick length-based pre-selection.
    
    Args:
        dali_splits (dict): split data of each subset. 
        subset (str): subset, e.g., "train" 

    Returns:
        dali_ids (list): segment ids.
        dali_nlen: note line length of each segment.
        dali_llen: sylphone line length of each segment.
    """
    
    dali_nlen = torch.stack([item[1] for item in dali_splits[subset][1]])
    dali_llen = torch.stack([item[2] for item in dali_splits[subset][1]])
    dali_ids = [item[0] for item in dali_splits[subset][1]]

    return dali_ids, dali_nlen, dali_llen

def pad_sequence(x: list,
                 y: list) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Pad a pair of sequences (x, y), each to the same length
    """

    len_x = [len(seq) for seq in x]
    len_y = [len(seq) for seq in y]
    
    # Pad the sequences  
    padded_x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)  
    padded_y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)  

    mask_x = torch.zeros((len(x), padded_x.shape[1]), device=padded_x.device).type(torch.bool)
    
    for k in range(len(x)):
        mask_x[k, len_x[k]:] = True

    mask_y = torch.zeros((len(y), padded_y.shape[1]), device=padded_y.device).type(torch.bool)
    
    for k in range(len(y)):
        mask_y[k, len_y[k]:] = True

    return padded_x, padded_y, mask_x, mask_y
    
def pytorch_cos_sim(a: torch.tensor, 
                    b: torch.tensor) -> torch.tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Args:
        a: 2D tensor of shape (m, d)
        b: 2D tensor of shape (n, d)

    Return: 
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """

    return cos_sim(a, b)

def cos_sim(a: torch.tensor, 
            b: torch.tensor) -> torch.tensor:
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Return: 
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """

    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)

    return torch.mm(a_norm, b_norm.transpose(0, 1))

def bresenham_line(m: int, 
                   n: int) -> list:
    """
    Estimate diagonal alignment between two sequences with lengths of m and n.

    Return:
        Alignment path with a list of index pairs.
    """

    diagonal_path = []
    
    # Starting point
    x, y = 0, 0
    
    # Ending point is the bottom-right corner of the grid
    x_end, y_end = n - 1, m - 1
    
    # Differences in x and y directions
    dx = abs(x_end - x)
    dy = abs(y_end - y)
    
    # Direction of the steps
    sx = 1 if x < x_end else -1
    sy = 1 if y < y_end else -1
    
    # Error values
    err = dx - dy
    
    while True:
        diagonal_path.append((y, x))
        
        # Check if we've reached the end point
        if x == x_end and y == y_end:
            break
        
        e2 = 2 * err
        
        if e2 > -dy:
            err -= dy
            x += sx
        
        if e2 < dx:
            err += dx
            y += sy
    
    return diagonal_path


def load_test_segs(test_dir: str,
                   dali_data: dict,
                   test_ids: list,
                   seg_lnum: int,
                   seg_overlap: int) -> tuple[list, list, list, list]:
    """
    Extracting note and sylphone features of test data and segmentation.

    Args:
        test_dir: path to test dir, i.e., DALI50.
        dali_data: pre-extracted dali data saved in HDF5 file, organized as
            "<song_id>":
                "note_encode" (torch.tensor): note features.
                "sylphone_encode" (torch.tensor): sylphone features.
                "sylphones" (list): each element is a sylphone (i.e., syllable vector).
        test_ids: list of song_ids in the test set.
        seg_lnum: number of lines in each segment.
        seg_overlap: number of overlapping lines between segments.

    Returns:
        seg_lines (list): each element is a segment, with information [seg_id, note feature, sylphone feature]. 
        seg_linelen (list): each element is a segment, with information [seg_id, note line length, sylphone line length]. 
        seg_path (list): each element is the reference alignment path for the segment: [seg_id, path matrix]
        seg_words (list): each element contains the word information of the segment: [seg_id, [words, word index]].
    """
    
    note_features_dali, syl_features_dali, path_truth, words_dali, sylphones_dali = load_test_data(test_dir, dali_data, test_ids)

    seg_lines = []
    seg_linelen = []
    seg_path = []
    seg_words = []
    for song_id in test_ids:
        note_seq = note_features_dali[song_id]
        syl_seq = syl_features_dali[song_id]
        path = path_truth[song_id]
        words = words_dali[song_id]

        words = [words[idx]+"," if idx in syl_seq[syl_seq[:,-1]==1, -2].int().tolist() else words[idx] for idx in range(len(words))]

        if sum(note_seq[:,-1] == 1) >= seg_lnum:
            note_line_idx = torch.cat((torch.tensor([-1]), torch.nonzero(note_seq[:,-1] == 1).squeeze(-1)))
            syl_line_idx = torch.cat((torch.tensor([-1]), torch.nonzero(syl_seq[:,-1] == 1).squeeze(-1)))

            assert len(note_line_idx) == len(syl_line_idx)

            note_lines = [note_seq[note_line_idx[k]+1:note_line_idx[k+1]+1] for k in range(len(note_line_idx)-1)]
            syl_lines = [syl_seq[syl_line_idx[k]+1:syl_line_idx[k+1]+1] for k in range(len(syl_line_idx)-1)]

            note_segs = [note_lines[k:k+seg_lnum] for k in range(len(note_lines)-seg_lnum+1)[::seg_lnum-seg_overlap]] # range for start, include last
            syl_segs = [syl_lines[k:k+seg_lnum] for k in range(len(syl_lines)-seg_lnum+1)[::seg_lnum-seg_overlap]]

            note_segs_llen = [torch.tensor([line.size(0) for line in item]) for item in note_segs]
            syl_segs_llen = [torch.tensor([line.size(0) for line in item]) for item in syl_segs]

            note_segs = [[note_encode(torch.cat(item, 0)).float(), torch.cat(item, 0).float()] for item in note_segs]
            syl_segs = [torch.cat(item, 0) for item in syl_segs]

            item_words = [[words[int(syl_segs[k][0, -2].item()): int(syl_segs[k][-1, -2].item())+1], syl_segs[k][:, -2] - syl_segs[k][0, -2]] for k in range(len(syl_segs))]

            seg_words.extend([[song_id+'_'+str(k), item_words[k]] for k in range(len(item_words))])

            syl_segs = [torch.from_numpy(np.delete(item.numpy(), -2, axis=-1)).float() for item in syl_segs]  # remove stopwords and word index
            seg_lines.extend([[song_id+'_'+str(k), note_segs[k][0], syl_segs[k], note_segs[k][1]] for k in range(len(note_segs))])
            seg_linelen.extend([[song_id+'_'+str(k), note_segs_llen[k], syl_segs_llen[k]] for k in range(len(note_segs_llen))])

            # get seg truth path
            syl_sec_starts = syl_line_idx[np.arange(len(syl_lines)-seg_lnum+1)[::seg_lnum-seg_overlap]] + 1
            syl_sec_ends = syl_line_idx[np.arange(len(syl_lines)+1)[::seg_lnum-seg_overlap]][1:]

            item_path = []
            for m in range(len(syl_sec_ends)):
                syl_start = torch.nonzero(path[:,1] == syl_sec_starts[m]).squeeze(0)[0].item()
                syl_end = torch.nonzero(path[:,1] == syl_sec_ends[m]).squeeze(0)[-1].item()
                item_path.append(path[syl_start:syl_end+1])

            assert len(item_path) == len(syl_segs)

            seg_path.extend([[song_id+'_'+str(k), item_path[k]] for k in range(len(item_path))])

    return seg_lines, seg_linelen, seg_path, seg_words

def load_test_data(test_dir: str,
                   dali_data: dict,
                   test_ids: list) -> tuple[list, list, list, list, list]:
    """
    Load test data, i.e., DALI50.

    Args:
        test_dir: path to test data, i.e., DALI50.
        dali_data: load HDF5 data of all DALI songs.
        test_ids: song ids of the test data.

    Returns:
        note_features_dali (dict): keys are song ids (str); values are the corresponding note features (torch.tensor).
        syl_features_dali (dict): keys are song ids; values are the corresponding sylphone features (torch.tensor).
        path_truth (dict): keys are song ids (str); values are the reference align path between notes and sylphones (torch.tensor).
        words_dali (dict): keys are song ids (str); values are the corresponding lyrics words (list).
        sylphones_dali (dict): keys are song ids (str); values are the corresponding lyrics sylphones (list).
    """

    syl_features_dali = {key: [] for key in test_ids}
    note_features_dali = {key: [] for key in test_ids}
    path_truth = {key: [] for key in test_ids}
    words_dali = {key: [] for key in test_ids}
    sylphones_dali = {key: [] for key in test_ids}

    for file_id in test_ids:

        file = os.path.join(test_dir, file_id+'_sylphone.txt')
        syls = pd.read_csv(file, sep='\t', header=None)
        notes = pd.read_csv(file.replace('_sylphone.txt', '_note.txt'), sep='\t', header=None)

        words = pd.read_csv(file.replace('_sylphone.txt', '_word.txt'), sep='\t', header=None)
        words = " ".join(list(words[2].values))
        words = clean_and_normalize_text(words)
        song_sylphone_encode, song_sylphones = sylphone_encode(words)
        sylphones_dali[file_id] = song_sylphones
        words_dali[file_id] = words.split(" ")

        note_features = dali_data[file_id]['note_encode'][:]

        if file_id == '3b5a09aff71d46c8890a095577080ba0':
            # Remove the error in this file (3b5a09aff71d46c8890a095577080ba0): remove the solo note in  
            # array([ 63.      , 121.304276, 142.78154 ,   1.      ,   0.      ,
            #  0.      ], dtype=float32)
            note_features = np.delete(note_features, 215, axis=0)

        note_features = torch.from_numpy(note_features)

        assert song_sylphone_encode.shape[0] == syls.shape[0]
        assert len(words.split(" ")) == song_sylphone_encode[-1, -1].item()+1

        syl_features_dali[file_id] = song_sylphone_encode
        # pitch, start, end, line_mark, stopwords, word index
        note_end = note_features[note_features[:, -1] == 1][:, 2]

        syls_end = torch.from_numpy(syls[1].values.copy())
        syl_line_marker = torch.stack([(abs(syls_end - item) == abs(syls_end - item).min()
                                    ).nonzero(as_tuple=True)[0][-1] for item in note_end])

        song_sylphone_encode = torch.cat((song_sylphone_encode, torch.zeros(len(song_sylphone_encode), 1)), -1)
        song_sylphone_encode[syl_line_marker, -1] = 1

        note_features_dali[file_id] = note_features
        syl_features_dali[file_id] = song_sylphone_encode

        syls = np.array(syls)
        notes = np.array(notes)

        file_path = []
        for m in range(len(syls)):
            note_start_idx = np.where(notes[:, 0] == syls[m, 0])[0][0]
            note_end_idx = note_start_idx
            while notes[note_end_idx , 1] < syls[m, 1]:
                note_end_idx  += 1

            file_path.extend([[i,m] for i in range(note_start_idx, note_end_idx+1)])
        path_truth[file_id] = torch.tensor(file_path)

    return note_features_dali, syl_features_dali, path_truth, words_dali, sylphones_dali


def define_metrics() -> Dict[str, Any]:
    """
    Create a dict to save the alignment and rhyme-related metrics.
    """

    metric_dict = {'multi-notes': [],
                    'multi-sylphones': [],
                    'rhyme density': [],
                    'rhyme distance': [],
                    'rhyme strength': [],
                    'longnote-stress': [],
                    'longnote-nonstop': [],
                    'longnote-longvowel': [],
                    }
    
    return metric_dict

def compute_stress_metrics(music: torch.tensor,
                    music_mask: torch.tensor, 
                    lyrics: torch.tensor, 
                    lyrics_mask: torch.tensor,
                    path: np.array) -> Dict[str, Any]:
    """
    Create a dict to save the alignment and rhyme-related metrics.
    """

    stress_extreme_dict = {
                            'longnote-stress': [],
                            'longnote-nonstop': [],
                            'longnote-longvowel': [],
                            'multi-notes': [],
                            'multi-sylphones': [],}
    

    vowel_idx, stress_idx, conso_end_idx, longvowel_idx = sylphone_feature_idx()

    # long note duration should be relative, not selected from absolute duration
    dur_idx = torch.argmax(music[:, 129:129+24], dim=-1)
    longnotes_idx = torch.nonzero(dur_idx >= torch.quantile(dur_idx.float(), 0.75)).squeeze()
    longnotes_idx = longnotes_idx.unsqueeze(0) if longnotes_idx.dim()==0 else longnotes_idx
    longnotes_idx = longnotes_idx.tolist()

    syl_feature = lyrics[~lyrics_mask]

    one_to_one_path = [[x, y] for x, y in path if Counter(path[:, 0])[x] == 1 and Counter(path[:, 1])[y] == 1]
    syl_idx_with_longnotes = [y for x, y in one_to_one_path if x in longnotes_idx]

    if len(longnotes_idx) > 0 and len(syl_idx_with_longnotes) > 0:
        longsyl_longvowel = torch.sum(torch.stack([syl_feature[idx, longvowel_idx] for idx in syl_idx_with_longnotes])) / (len(syl_idx_with_longnotes) + 1e-6)
        longsyl_stress = torch.sum(torch.stack([syl_feature[idx, stress_idx[1:]] for idx in syl_idx_with_longnotes])) / (len(syl_idx_with_longnotes) + 1e-6)
        longsyl_nonstop = torch.sum(torch.stack([syl_feature[idx, -2] == 0 for idx in syl_idx_with_longnotes])) / (len(syl_idx_with_longnotes) + 1e-6)

    else:
        longsyl_longvowel = torch.tensor(0)
        longsyl_stress = torch.tensor(0)
        longsyl_nonstop = torch.tensor(0)
    
    stress_extreme_dict['longnote-stress'] = longsyl_stress.item()
    stress_extreme_dict['longnote-nonstop'] = longsyl_nonstop.item()
    stress_extreme_dict['longnote-longvowel'] = longsyl_longvowel.item()

    stress_extreme_dict['multi-notes'] = Counter(path[:,1]).most_common()[0][1]
    stress_extreme_dict['multi-sylphones'] = Counter(path[:,0]).most_common()[0][1]

    return stress_extreme_dict


def compute_rhyme_metrics(lyrics, 
                          lyrics_mask) -> Dict[str, Any]:
    """
    Compte rhyme-related metrics: rhyme density, strength, and distance.
    """

    # parameters
    vowel_idx, stress_idx, conso_end_idx, longvowel_idx = sylphone_feature_idx()

    fullsyl_idx = torch.cat((vowel_idx, stress_idx, 
                            conso_end_idx), 0)
    rhyme_idx = torch.cat((vowel_idx, conso_end_idx), 0)

    # rhyme reference
    truth_sylline = lyrics[-1][~lyrics_mask[-1]].cpu()
    truth_syl_vowel = truth_sylline[truth_sylline[:,-1]==1][:, vowel_idx]
    truth_ssm_vowel = pytorch_cos_sim(truth_syl_vowel, truth_syl_vowel)
    truth_syl_rhyme = truth_sylline[truth_sylline[:,-1]==1][:, rhyme_idx]

    truth_syl_fullsyl = truth_sylline[truth_sylline[:,-1]==1][:, fullsyl_idx]
    truth_ssm_fullsyl = pytorch_cos_sim(truth_syl_fullsyl, truth_syl_fullsyl)
    truth_ssm_repeat = truth_ssm_fullsyl.clone()
    truth_ssm_repeat[abs(truth_ssm_repeat - 1.0) > 1e-2] = 0 


    # rhyme of candidates
    rhyme_dict = {
                    'rhyme density': [],
                    'rhyme distance': [],
                    'rhyme strength': [],
                    }

    y_top = lyrics[0][~lyrics_mask[0]]
    item = y_top
    syl_vowel = item[item[:,-1]==1][:, vowel_idx]
    ssm_vowel = pytorch_cos_sim(syl_vowel, syl_vowel)
    syl_rhyme = item[item[:,-1]==1][:, rhyme_idx]

    syl_fullsyl = item[item[:,-1]==1][:, fullsyl_idx]
    ssm_fullsyl = pytorch_cos_sim(syl_fullsyl, syl_fullsyl)

    ssm_repeat = ssm_fullsyl.clone()
    ssm_repeat[abs(ssm_repeat - 1.0) > 1e-2] = 0 

    assert ssm_vowel.size(0) == truth_ssm_vowel.size(0)

    rhyme_pos_candi, rhyme_counts_candi = rhyme_pos_compute(syl_vowel)
    rhyme_pos_truth, rhyme_counts_truth = rhyme_pos_compute(truth_syl_vowel)

    rhyme_density = len(rhyme_pos_candi) / (len(syl_vowel) + 1e-6)

    unique_rhyme_vowel = torch.unique(syl_vowel[rhyme_pos_candi], dim=0)
    rhyme_pervowel_diversity = torch.mean(torch.stack([len(torch.unique(syl_rhyme[torch.nonzero((syl_vowel==item).all(dim=1)).squeeze(), :], dim=0)) /
                               torch.sum((syl_vowel==item).all(dim=1)) for item in unique_rhyme_vowel])).item() if len(unique_rhyme_vowel) > 0 else 0

    rhyme_vowel_diversity = len(torch.unique(syl_vowel[rhyme_pos_candi], dim=0)) / (len(rhyme_pos_candi) + 1e-6)

    activate_candi = F.one_hot(rhyme_pos_candi, num_classes=len(syl_vowel)).sum(0)
    activate_truth = F.one_hot(rhyme_pos_truth, num_classes=len(syl_vowel)).sum(0)

    activate_union = torch.logical_or(activate_candi, activate_truth)

    rhyme_distance_pattern = sum(abs(activate_candi - activate_truth)) / (sum(activate_union) + 1e-6)
    rhyme_distance = rhyme_distance_pattern # + rhyme_distance_vowel
    
    rhyme_dict['rhyme density'] = rhyme_density
    rhyme_dict['rhyme strength'] = (1 -rhyme_vowel_diversity + 1 - rhyme_pervowel_diversity) / 2
    rhyme_dict['rhyme distance'] =  rhyme_distance.item()

    return rhyme_dict

def rhyme_pos_compute(syl_vowel: torch.tensor) -> tuple:
    """
    Extract rhyming positions.

    """

    rhyme_pos = []
    rhyme_idx = torch.unique(syl_vowel, dim=0, return_inverse=True, return_counts=True)[1]
    _, counts = rhyme_idx.unique(return_counts=True)
    repeated = rhyme_idx.unsqueeze(0) == rhyme_idx.unique()[counts - 1 > 0].unsqueeze(1)
    rhyme_pos = repeated.any(0).nonzero(as_tuple=True)[0]

    rhyme_counts = counts - 1 
    rhyme_counts = torch.sum(rhyme_counts[rhyme_counts>0])

    return rhyme_pos, rhyme_counts

def output_alignment(lyrics, 
                     path, 
                     melody_ori, 
                     id_pair, 
                     path_save, 
                     words, 
                     rank=None) -> None:
    """
    Output alignment between notes and words.
    """

    if rank is None:
        rank = 'truth'
    else:
        rank = str(rank)
    name = id_pair[0] + '_' + rank + '_'+ id_pair[1]

    phone_list, vowels, stresses, consonants = get_phone_dict()
    
    le = LabelEncoder()  

    # save the start and end time of each sylphone according note-syl align path
    syl_times = []
    for idx in np.unique(path[:,1]):
        item = {"time": [], "text": []}
        # start and end time of sylphone
        start_note_idx = path[path[:,1] == idx, :][0,0]
        end_note_idx = path[path[:,1] == idx][-1, 0]
        syl_start = melody_ori[start_note_idx, 1].item() - melody_ori[0, 1].item() # starting note start
        syl_end = melody_ori[end_note_idx, 2].item() - melody_ori[0, 1].item() # ending note end
        # sylphone content
        # lyrics 68: 66 sylphone, nonstop, line
        decoded = le.inverse_transform(torch.nonzero(lyrics[idx, :-2]).squeeze())
        # sylphone content
        vowel = [item for item in decoded if item in vowels]
        stress = [item for item in decoded if item in stresses]
        end_cons = [item for item in decoded if item in consonants]
        sylphone = ' '.join([''.join(vowel+stress)] + end_cons)
        item['time'] = [syl_start, syl_end]
        item['text'] = sylphone
        syl_times.append(item)
    # save into .txt file
    dali_code.write_annot_txt(syl_times, name+'_sylphones', path_save)

    assert id_pair[1] == words[0] or id_pair[1] == words[0].replace('_shuffled', '') \

    line_start_idx = torch.nonzero(melody_ori[:, -1] == 1).squeeze()[:-1] + 1 
    line_start_idx = torch.tensor([0] + line_start_idx.tolist())
    line_end_idx =  torch.nonzero(melody_ori[:, -1] == 1).squeeze()
    line_starts = melody_ori[line_start_idx, 1] - melody_ori[0, 1]
    line_starts = line_starts.tolist()
    line_ends = melody_ori[line_end_idx, 2] - melody_ori[0, 1]
    line_ends = line_ends.tolist()

    words_file = id_pair[0] + '_' + rank + '_'+ id_pair[1] + '_words'
    words_text = words[1][0] if words[1] is not None else 'shuffled'

    syl_time_list = [item['time'] for item in syl_times]

    word_times = []

    if words_text == 'shuffled':
        # if words is shuffled, then use the sylphone time to save the word time
        word_start = syl_time_list[0][0] 
        word_end = syl_time_list[-1][-1] 
        word_times.append({"time": [word_start, word_end], "text": words_text})
    else:
        for k in range(len(words_text)):
            word_idx = torch.unique(words[1][1])[k]
            item_idx = torch.nonzero(words[1][1] == word_idx).squeeze().tolist() 
            word_start = syl_time_list[item_idx[0]][0] if isinstance(item_idx, list) else syl_time_list[item_idx][0]
            word_end = syl_time_list[item_idx[-1]][-1] if isinstance(item_idx, list) else syl_time_list[item_idx][-1]
            word_times.append({"time": [word_start, word_end], "text": words_text[k]})

    dali_code.write_annot_txt(word_times, words_file, path_save)