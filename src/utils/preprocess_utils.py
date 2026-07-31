from collections import defaultdict
import librosa
from g2p_en import G2p
from nltk.corpus import stopwords
from num2words import num2words
import os
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import torch
import torch.nn.functional as F

from src.utils import syllabifier

stops = set(stopwords.words('english'))  
g2p = G2p()
language = syllabifier.English  # syllabifier.loadLanguage("english.cfg")


def remove_repeats(lst: list) -> list:
    """
    Remove repeated elements from a list while keeping the first occurrence.

    Args:
        lst (list): The input list from which to remove repeated elements.

    Returns:
        list: A new list with repeated elements removed, keeping the first occurrence.
    """
    # Create a dictionary to map each element to its indices
    index_map = defaultdict(list)
    for i, val in enumerate(lst):
        index_map[val].append(i)
    
    # Filter to only repeated elements
    repeats = {key: indices for key, indices in index_map.items() if len(indices) > 1}

    repeats = [repeats[key][:-1] for key in repeats.keys()]
    repeats = [item for sublist in repeats for item in sublist]  # flatten the list
    idx_keep = [k for k in range(len(lst)) if k not in repeats]  # remove repeated lines

    return idx_keep

def melody_feature_extract(notes: list, 
                           lines: list) -> torch.Tensor:
    """
    Extract melody features from the notes and lines of a song.

    Args:
        notes (list): List of note annotations.
        lines (list): List of line annotations.

    Returns:
        torch.Tensor: Melody feature tensor.
    """
    # note features

    lines = [line for line in lines if clean_and_normalize_text(line['text']) != '']  # filter out empty lines
    lines = [line for line in lines if line['time'][0] >= 0]  # filter out empty lines
    notes = [note for note in notes if note['time'][0] >= lines[0]['time'][0]]  # filter out empty notes

    note_seq = []
    notes_start_end = [(note['time'][0], note['time'][1]) for note in notes]
    lines_start_end = [(line['time'][0], line['time'][-1]) for line in lines]

    notes_idx_keep = remove_repeats(notes_start_end)  # remove repeated notes
    lines_idx_keep = remove_repeats(lines_start_end)  # remove repeated lines

    notes = [notes[k] for k in notes_idx_keep]
    lines = [lines[k] for k in lines_idx_keep]

    line_ends = [line['time'][-1] for line in lines]

    note_ends = []
    for note in notes:
        pitch = int(librosa.hz_to_midi(note["freq"][0]))
        start = note['time'][0]
        end = note['time'][1]
        if end in line_ends and end not in note_ends:
            line_mark = 1
            note_ends.append(end)
        else:
            line_mark = 0

        note_seq.append([pitch, start, end, line_mark])

    note_seq = torch.tensor(note_seq)

    # ensure all lines are marked
    try:
        assert sum(note_seq[:, -1] == 1).item() == len(lines)
    except AssertionError:
        print("music and lyrics line mismatch")

    return note_seq

def lyrics_feature_extract(lines: list) -> tuple[torch.Tensor, list]:
    """
    Extract syllable features from the lines of a song.

    Args:
        lines (list): List of line annotations.

    Returns:
        torch.Tensor: Syllable feature tensor.
    """

    lines = [line for line in lines if clean_and_normalize_text(line['text']) != '']  # filter out empty lines
    
    lines = [line for line in lines if line['time'][0] >= 0]  # filter out empty lines
    lines_start_end = [(line['time'][0], line['time'][-1]) for line in lines]
    lines_idx_keep = remove_repeats(lines_start_end)  # remove repeated lines
    lines = [lines[k] for k in lines_idx_keep]

    syl_seq, sylphones = [], []
    for line in lines:
        line_text = line['text']
        line_text = clean_and_normalize_text(line_text)
        line_sylseq, line_sylphones = sylphone_encode(line_text)

        # add line marker at the end of the sequence
        line_marker = torch.zeros((len(line_sylseq), 1))
        line_marker[-1, -1] = 1
        line_sylseq = torch.cat((line_sylseq, line_marker), -1) 

        syl_seq.append(line_sylseq) 
        sylphones.append(line_sylphones)

    syl_seq = torch.cat(syl_seq, 0)

    sylphones = [item for sublist in sylphones for item in sublist]  # flatten the list

    return syl_seq, sylphones

def get_phone_dict() -> tuple[list, list, list, list]:
    """
    Get the phone dictionary for syllable encoding.
    
    Returns:
        Tuple: A tuple containing the phone dictionary, vowel base, vowel stress, and consonants.
    """

    # CMU Pronunciation Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
    vowel_base = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
    vowel_stress = ['0', '1', '2']
    consonants = ['B', 'CH', 'D', 'DH' , 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 
                'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'] 

    phone_dict = vowel_base + vowel_stress + consonants # 15+3+24=42

    return phone_dict, vowel_base, vowel_stress, consonants

def sylphone_feature_idx() -> tuple[list, ]:
    """
    Get the indices for syllable features.

    Returns:
        A tuple containing the indices for vowels, stresses, conda consonants, and long vowels.
    """
    
    phone_list, vowels, stresses, consonants = get_phone_dict()
    
    le = LabelEncoder()  
    label_encode = le.fit_transform(phone_list)
    label_mapping = dict(enumerate(le.classes_))
    label_encode = torch.from_numpy(label_encode)

    conda_conso_idx = torch.tensor([[key for key, val in label_mapping.items()
                                if val == value][0] for value in consonants])
    vowel_idx = torch.tensor([[key for key, val in label_mapping.items()
                                if val == value][0] for value in vowels])
    stresses = ['0', '1', '2']
    stress_idx = torch.tensor([[key for key, val in label_mapping.items()
                                if val == value][0] for value in stresses])

    # long vowels: AA, AO, AW, AY, EY, IY, OW, OY, UW
    # Reference for this: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/ismir2009MelodyAndLyrics.pdf
    longvowels = ['AA', 'AO', 'AW', 'AY', 'EY', 'IY', 'OW', 'OY', 'UW']
    longvowel_idx = torch.tensor([[key for key, val in label_mapping.items()
                                if val == value][0] for value in longvowels])

    return vowel_idx, stress_idx, conda_conso_idx, longvowel_idx

def sylphone_encode(input_words: list) -> tuple[torch.Tensor, list]:
    """
    Convert input words to syllable-level phoneme sequences (sylphones/syllabel vectors).

    Args:
        input_words: list of words to be converted. Each element is a word.

    Returns:
        song_sylphone_encode: torch.Tensor, sylphone-wise feature.
        song_sylphones: list of sylphones.
    """

    # get phone dictionary
    phone_list, vowels, stress, consonants = get_phone_dict()

    # le.classes_ follow alphabetic order (deterministic):
    le = LabelEncoder()  
    on = OneHotEncoder(sparse_output=False)

    le.fit_transform(phone_list)
    on.fit_transform([phone_list])

    # words to pheneme sequence
    input_words = input_words.split(" ")  # convert to uppercase
    input_words = [word_norm(word) for word in input_words]  # normalize words 
    song_phone_seq = [' '.join(g2p(word)) for word in input_words]

    # get sylphone for each line
    song_sylphone_encode = []
    song_sylphones = []

    # get sylphone for each line
    word_idx, stopword_index = [], []
    for k in range(len(song_phone_seq)):
        word = song_phone_seq[k]
        # separate phoneme sequence into syllable-level phonemes (sylphone)
        word = syllabifier.stringify(syllabifier.syllabify(language, word))
        word = word.split(' . ')
             
        # covert each sylphone into a one-hot vector, activated by all phonemes it contains
        for syllable in word:
            stress_level = list(set(syllable).intersection('012'))[0]
            syllable_nostress = syllable.replace(stress_level, '')
            sylphone_items = syllable_nostress.split(' ')
            vowel = set(sylphone_items).intersection(vowels)
            vowel_idx = sylphone_items.index(list(vowel)[0])

            conso_beforevowel = sylphone_items[:vowel_idx]
            conso_aftervowel = sylphone_items[vowel_idx:]

            cons_front = [item for item in conso_beforevowel if item in consonants]
            cons_end = [item for item in conso_aftervowel if item in consonants]

            sylphone_items = list(vowel) + [stress_level] + cons_end
            syll_phone_seq = torch.from_numpy(le.transform(sylphone_items))
            y_onehot = F.one_hot(syll_phone_seq, num_classes=len(phone_list))
            y_onehot = y_onehot.sum(dim=0)
            song_sylphone_encode.append(y_onehot)
            sylphone_items = cons_front + [list(vowel)[0] + stress_level] + cons_end
            song_sylphones.append(' '.join(sylphone_items))
        
        # song_sylphone_ori.extend(word)
        word_idx.extend([k] * len(word))
        stopword_index.extend([1 if input_words[k].lower() in stops else 0] * len(word))

    song_sylphone_encode = torch.cat((
        torch.stack(song_sylphone_encode), 
        torch.tensor(stopword_index).unsqueeze(1),
        torch.tensor(word_idx).unsqueeze(1)), -1)

    # sylphone vectors for a song
    return song_sylphone_encode, song_sylphones


def year_to_pronunciation(year: int | str) -> str:
    """Convert a year to its pronunciation in words.

    Args:
        year: The year to convert.

    Returns:
        str: The pronunciation of the year in words.
    """

    year = int(year)
    if 1000 <= year < 2000:
        # Handle years from 1000 to 1999
        return f"{num2words(year // 100)} {num2words(year % 100)}".replace(" and ", " ").strip()
    elif 2000 <= year < 2100:
        # Handle years from 2000 to 2099
        return f"two thousand {num2words(year % 100)}".replace(" and ", " ").strip()
    else:
        # Handle years outside of 1000-2099 if needed
        return year

def clean_and_normalize_text(phrase: str) -> str:
    """
    Clean and standardize general text.
    """

    phrase = phrase.replace('$', 'dollar').replace('€', 'euro'
                    ).replace('£', 'pound').replace('huah', 'ah').replace('Mmmm', 'Em'
                    ).replace('mmmm', 'em').replace('Shh', 'Shi').replace('shh', 'shi'
                    ).replace('&', 'and').replace(',', ' ')
    
    phrase = re.sub(r"\b [b-z] \b", lambda match: match.group().upper(), phrase)

    # Remove special while keeping alphanumeric, spaces, and apostrophes
    phrase = re.sub(r"(?<!\w)'|'(?!\w)", '', phrase)

    def replacement_function(match):
        value = match.group()
        if value.isdigit():
            year = int(value)
            if 1000 <= year <= 2099:
                return year_to_pronunciation(year)
            else:
                return num2words(year)
        return '' 

    # Regular expression to match digits and special characters
    cleaned = re.sub(
        r"\d+", 
        replacement_function, 
        phrase
    )

    cleaned = re.sub(r"[^a-zA-Z0-9\s']", '', cleaned)

    normalized = re.sub(r'\s+', ' ', cleaned).strip()

    return normalized

def word_norm(text: str) -> str:
    """
    Normalize words by converting to uppercase and handling ordinal numbers.
    """

    ordinal_num = ['1st', '2nd', '3rd'] + [str(k)+'th' for k in range(4,100)] + \
                    [str(k) for k in range(100)]
    ordinal_num = [item.upper() for item in ordinal_num]
    ordinal_word = [num2words(k, ordinal=True).upper() for k in range(1, 100)] + \
                    [num2words(k, ordinal=False).upper() for k in range(0, 100)]
    
    for k in range(len(text)):
        if text[k].upper() in ordinal_num:
            text[k] = ordinal_word[ordinal_num.index(text[k].upper())]

    return text

def note_encode(input_notes: torch.tensor) -> torch.tensor:
    """
    Convert note annotations into a feature tensor.

    Args:
        input_notes: A tensor containing note annotations with shape (N, 4) where N is the number of notes.
            Each note is represented as [pitch, start, end, line marker]. Note that line marker is only used
            for segmentation purpose.

    Returns:
        A tensor of shape (N, 178) containing the encoded 177D note features and one binary value indicating
            line marker.
    """

    # pitch, abosulte start, end, line_mark => pitch, pitch change, duration, onset_shift, line_mark
    durations = input_notes[:,2] - input_notes[:, 1]  # end - start
    onset_shift = input_notes[1:, 1] - input_notes[:-1, 1]  # start - previous start
    input_notes[:, 1] = durations
    input_notes[1:, 2] = onset_shift
    input_notes[0, 2] = 0

    # pitch change normalized to the first note
    pitch_change = input_notes[:, 0].unsqueeze(-1) - input_notes[0, 0].unsqueeze(-1)
    # pitch change to one-hot
    sign_change = pitch_change.float().clone()
    sign_change[sign_change >= 0] = 1
    sign_change[sign_change < 0] = 0
    pitch_change = F.one_hot(abs(pitch_change).long(), num_classes=128).float().squeeze()

    dur_rest = input_notes[:, 1:3]
    dur_rest = torch.log2(dur_rest + 1e-6)
    dur_rest = (dur_rest - torch.min(dur_rest, 0, keepdim=True).values) / (torch.max(dur_rest, 0, keepdim=True).values
                         - torch.min(dur_rest, 0, keepdim=True).values + 1e-6)
    dur_rest[dur_rest < 0] = 0

    dur_onehot = discretize_to_onehot(dur_rest[:, 0], num_units=24)
    rest_onehot = discretize_to_onehot(dur_rest[:, 1], num_units=24)

    note_seq = torch.cat((pitch_change, sign_change, dur_onehot, rest_onehot, input_notes[:, -1].unsqueeze(-1)), -1)

    return note_seq
    

def discretize_to_onehot(values: torch.Tensor, 
                         num_units: int=12) -> torch.Tensor:
    """
    Convert a sequence of continuous values in [0,1] into one-hot encoded tensors.

    Args:
        values: A tensor of continuous values in [0,1].
        num_units: The number of discrete units for one-hot encoding.

    Returns:
        A one-hot encoded tensor of shape (N, num_units) where N is the length of the input values.
    """

    # Ensure values are within [0,1]
    values = torch.clamp(values, 0, 1)

    # Compute indices, ensure they are within valid range
    indices = (values * (num_units - 1)).long()
    indices = torch.clamp(indices, 0, num_units - 1)  # Prevent out-of-range errors

    # Ensure indices are int64 (required by F.one_hot)
    indices = indices.to(dtype=torch.int64)

    # Use F.one_hot for efficient one-hot encoding
    one_hot = F.one_hot(indices, num_classes=num_units).float()

    return one_hot

def save_notes(song_notes: torch.Tensor, save_path: str) -> None:
    """
    Save note annotations to a text file.

    Args:
        song_notes: A tensor containing note annotations with shape (N, 3) where N is the number of notes.
            Each note is represented as [pitch, start, end].
        save_path: The path to save the note annotations.
    """

    note_num = len(song_notes)

    notes = []
    for k in range(note_num):
        # freq_to_pitch
        note_pitch = song_notes[k][0].int().item()
        # Create a Note instance with start, end, pitch, line marker
        start = song_notes[k][1].item() - song_notes[0][1].item()
        end = song_notes[k][2].item() - song_notes[0][1].item()
        notes.append([note_pitch, start, end])

    with open(save_path, 'w') as f:
        for note in notes:
            f.write(f"{note[0]}\t{note[1]}\t{note[2]}\n")