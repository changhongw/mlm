from glob import glob
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.constrained_layout.use'] = True
import numpy as np
import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def visual_lyrics(words, l_plot, ax, idx, color, alpha=0.9):
    """ Visualize the lyrics words or sylphones in the plot.
    Args:
        words (pd.DataFrame): DataFrame containing 'start', 'end', and 'label' columns.
        l_plot (int): Number of words to plot.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        idx (int): Index of the subplot.
        color (str): Color for the background of the intervals.
        alpha (float): Transparency level for the background color.
    """

    visual = words[:l_plot].values

    # Plot vertical lines for each start and end of the intervals
    for start, end, label in visual:
        ax[idx].axvspan(start, end, color=color, alpha=alpha)
        ax[idx].axvline(x=start, color='gray', linestyle=':')  # Start of the interval
        ax[idx].axvline(x=end, color='gray', linestyle=':')    # End of the interval

        # Calculate the midpoint of the interval
        midpoint = (start + end) / 2
        
        # Place the label at the midpoint
        ax[idx].text(midpoint, 0.5, label.replace(' ', '\n'), ha='center', va='center') # .replace(',', '').replace('.', '')

    # Adjust the plot limits and labels
    ax[idx].set_ylim(0, 1)  # Set y-axis limits
    ax[idx].set_yticks([])

def dataframe_fingerprint(df: pd.DataFrame) -> int:
    return pd.util.hash_pandas_object(df, index=True).sum()

def correct_overlap_annotations(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(by=["start", "end"]).reset_index(drop=True)

    # boundaries
    bounds = sorted(set(df["start"]) | set(df["end"]))

    segments = []

    # build segments with labels
    for s, e in zip(bounds, bounds[1:]):
        mask = (df["start"] <= s) & (df["end"] >= e)
        candidates = df[mask]

        if not candidates.empty:
            # Pick the most specific interval (shortest duration)
            best = candidates.iloc[(candidates["end"] - candidates["start"]).argmin()]
            segments.append((s, e, best["label"]))

    return pd.DataFrame(segments, columns=["start", "end", "label"])

test_outputs = "../datasets/test_alignment_output/"
mlm_cal = "seg4_mlmcal_top11_gamma0.1_a0.25"

note_files = glob(os.path.join(test_outputs, mlm_cal, "*_notes.txt"))[::-1]
seg_ids = [os.path.basename(f).replace('_notes.txt', '') for f in note_files]

seg_id = "109d0e9f08e54395b679a4600c11d388_10"
seg = seg_id
note_file = os.path.join(test_outputs, mlm_cal, seg + '_notes.txt')

# Load the MIDI file
notes = pd.read_csv(note_file, sep='\t', header=None).values
starts = notes[:, 1]
ends = notes[:, 2]
missing_ends = np.array([end for end in ends if end not in starts])
pitch_max = notes[:, 0].max()
pitch_min = notes[:, 0].min()

words_truth = glob(os.path.join(test_outputs, mlm_cal, f"{seg}_truth*_words.txt"))
num = 0
while num < 11:
    f = glob(os.path.join(test_outputs, mlm_cal, f"{seg}_{num}_*_words.txt"))[0]
    base_name = os.path.basename(f)
    middle = base_name[len(f"{seg}_{num}_"):-len("_words.txt")]
    if "shuffled" not in f:     
        words_cal = glob(os.path.join(test_outputs, mlm_cal, f"{seg}_{num}_*_words.txt"))    
        break
    else:
        num += 1

words_length = glob(os.path.join(test_outputs, "seg4_lengthInform_topk1", f"{seg}_*_words.txt"))
words_random = glob(os.path.join(test_outputs, "seg4_random_topk1", f"{seg}_*_words.txt"))

words_truth = pd.read_csv(words_truth[0], sep='\t', header=None, names=['start', 'end', 'label'])
words_truth = words_truth.groupby(['start', 'end'], as_index=False).agg({'label': ' '.join})

words_cal = pd.read_csv(words_cal[0], sep='\t', header=None, names=['start', 'end', 'label'])
words_cal = words_cal.groupby(['start', 'end'], as_index=False).agg({'label': ' '.join})

words_length = pd.read_csv(words_length[0], sep='\t', header=None, names=['start', 'end', 'label'])
words_length = words_length.groupby(['start', 'end'], as_index=False).agg({'label': ' '.join})

words_random = pd.read_csv(words_random[0], sep='\t', header=None, names=['start', 'end', 'label'])
words_random = words_random.groupby(['start', 'end'], as_index=False).agg({'label': ' '.join})
# if overlap only ends, replace the end of the previous one with the end of the next one, to make it non-overlapping
words_random = correct_overlap_annotations(words_random)

time_min = starts.min() - 0.1
time_threshold = ends.max() + 0.1
# only keep first 5 seconds
words_truth = words_truth[words_truth['end'] < time_threshold]
words_cal = words_cal[words_cal['end'] < time_threshold]
words_length = words_length[words_length['end'] < time_threshold]
words_random = words_random[words_random['end'] < time_threshold]

colors = ["#fcf6b2", "#eed6e5", "#e1f0fa", "#CCFB5D"]
fig, ax = plt.subplots(9, 1, sharex=True, figsize=(12, 7.2),
                    gridspec_kw={'height_ratios': [4, 1, 1, 1, 1, 1.4, 1.4, 1.4, 2.2]})
alpha = [0.9, 0.9, 0.9, 0.5]

###### plot melody notes ######
# Plot each note as a horizontal bar (piano roll)
for pitch, start, end in notes:
    # ax[0].plot([start+0.04, end-0.04], [pitch, pitch],color="#22a1a9", lw=6)
    ax[0].plot([start+0.025, end-0.025], [pitch, pitch],color="#22a1a9", lw=6) 
    ax[0].axvline(x=start, color='gray', linestyle=':')    # End of the interval
    if end in missing_ends:
        ax[0].axvline(x=end, color='gray', linestyle=':')    # Start of the interval

note_end = end
ax[0].set_ylim([pitch_min-1.5, pitch_max+1.5])
ax[0].set_title("Melody query (MIDI)", loc='center', fontweight='bold')
ax[0].set_xlim([time_min, time_threshold])  # Set x-axis limit based on the last note's end time])


###### plot lyrics sylphones ######
k = 1
visual_lyrics(words_truth, len(words_truth), ax, k, color=colors[k-1], alpha=alpha[k-1])
ax[k].yaxis.set_label_coords(-0.017, 0.5)
lyrics_text = " ".join(words_truth["label"])
ax[k].set_xlim([time_min, time_threshold])
ax[k].set_ylabel('Ref', rotation='horizontal', va="center", fontweight='bold') 
ax[k].text(0.5, 1.15, "Lyrics words", transform=ax[k].transAxes, ha="center", va="bottom", fontweight='bold', fontsize=12)

k = 2
visual_lyrics(words_cal, len(words_cal), ax, k, color=colors[k-1], alpha=alpha[k-1])
ax[k].yaxis.set_label_coords(-0.017, 0.5)
lyrics_text = " ".join(words_cal["label"])
ax[k].set_xlim([time_min, time_threshold])
ax[k].set_ylabel('MC', rotation='horizontal', va="center", fontweight='bold')

k = 3
visual_lyrics(words_length, len(words_length), ax, k, color=colors[k-1], alpha=alpha[k-1])
ax[k].yaxis.set_label_coords(-0.017, 0.5)
lyrics_text = " ".join(words_length["label"])
ax[k].set_xlim([time_min, time_threshold])
ax[k].set_ylabel('LI', rotation='horizontal', va="center", fontweight='bold')

k = 4
visual_lyrics(words_random, len(words_random), ax, k, color=colors[k-1], alpha=alpha[k-1])
ax[k].yaxis.set_label_coords(-0.017, 0.5)
lyrics_text = " ".join(words_random["label"])
ax[k].set_xlim([time_min, time_threshold])
ax[k].set_ylabel('Ra', rotation='horizontal', va="center", fontweight='bold')


###### plot lyrics sylphones ######
syl_truth = glob(os.path.join(test_outputs, mlm_cal, f"{seg}_truth*_sylphones.txt"))
syl_cal = glob(os.path.join(test_outputs, mlm_cal, f"{seg}_{num}_*_sylphones.txt"))
syl_length = glob(os.path.join(test_outputs, "seg4_lengthInform_topk1", f"{seg}_*_sylphones.txt"))
syl_random = glob(os.path.join(test_outputs, "seg4_random_topk1", f"{seg}_*_sylphones.txt"))

syl_truth = pd.read_csv(syl_truth[0], sep='\t', header=None, names=['start', 'end', 'label'])
syl_cal = pd.read_csv(syl_cal[0], sep='\t', header=None, names=['start', 'end', 'label'])
syl_length = pd.read_csv(syl_length[0], sep='\t', header=None, names=['start', 'end', 'label'])
syl_random = pd.read_csv(syl_random[0], sep='\t', header=None, names=['start', 'end', 'label'])
syl_random = syl_random.groupby(['start', 'end'], as_index=False).agg({'label': ' '.join})

# only keep first 5 seconds
syl_truth = syl_truth[syl_truth['end'] < time_threshold]
syl_cal = syl_cal[syl_cal['end'] < time_threshold]
syl_length = syl_length[syl_length['end'] < time_threshold]
syl_random = syl_random[syl_random['end'] < time_threshold]

k = 5
visual_lyrics(syl_truth, len(syl_truth), ax, k, color=colors[k-5], alpha=alpha[k-5])
ax[k].set_ylabel('Ref', rotation='horizontal', labelpad=5, fontweight='bold')
ax[k].yaxis.set_label_coords(-0.015, 0.35)
ax[k].set_xlim([time_min, time_threshold])
ax[k].text(0.5, 1.15, "Lyrics syllables", transform=ax[k].transAxes, ha="center", va="bottom", fontweight='bold', fontsize=12)

k = 6
visual_lyrics(syl_cal, len(syl_cal), ax, k, color=colors[k-5], alpha=alpha[k-5])
ax[k].set_ylabel('MC', rotation='horizontal', labelpad=5, fontweight='bold')
ax[k].yaxis.set_label_coords(-0.015, 0.35) 
ax[k].set_xlim([time_min, time_threshold])

k = 7
visual_lyrics(syl_length, len(syl_length), ax, k, color=colors[k-5], alpha=alpha[k-5])
ax[k].set_ylabel('LI', rotation='horizontal', labelpad=5, fontweight='bold')
ax[k].yaxis.set_label_coords(-0.015, 0.35)  # x = left of axis, y = vertically centered
ax[k].set_xlim([time_min, time_threshold])

k = 8
visual_lyrics(syl_random, len(syl_random), ax, k, color=colors[k-5], alpha=alpha[k-5])
ax[k].set_ylabel('Ra', rotation='horizontal', labelpad=5, fontweight='bold')
ax[k].yaxis.set_label_coords(-0.015, 0.35)  # x = left of axis, y = vertically centered
ax[k].set_xlim([time_min, time_threshold])

ax[-1].set_xlabel("Time (s)")

plt.savefig('outputs/alignment_results_{}_syl.pdf'.format(seg))
# plt.savefig('outputs/alignment_results_{}_syl.png'.format(seg))
plt.close(fig)