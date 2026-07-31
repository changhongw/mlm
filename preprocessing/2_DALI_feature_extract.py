import DALI as dali_code
import h5py
import os
import pandas as pd
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.preprocess_utils import melody_feature_extract, lyrics_feature_extract

dali_data_path = '../data/DALI/DALI_v2.0/annot_tismir'
dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])
song_ids = list(dali_data.keys())

save_dir = '../data/'
df = pd.read_csv(os.path.join(save_dir, './DALI_English_songs.csv'))
dali_ids = df['English_song_ids'].tolist()

save_path = os.path.join(save_dir, "DALI_features.hdf5")
if os.path.exists(save_path):
    os.remove(save_path)

def main():
    feature_num = 0
    with h5py.File(save_path, "w") as f:
        buffer = []
        BATCH_SIZE = 64

        for id in tqdm(dali_ids):
            entry = dali_data[id]
            anno = entry.annotations['annot']

            try:
                note_seq = melody_feature_extract(anno['notes'], anno['lines'])
                syl_seq, sylphones = lyrics_feature_extract(anno['lines'])

                note_seq_np = note_seq.detach().cpu().numpy()
                syl_seq_np = syl_seq.detach().cpu().numpy()

                buffer.append((id, note_seq_np, syl_seq_np, sylphones))

            except Exception as e:
                print(f"{id}, {e}")
                continue

            feature_num += 1

            if len(buffer) >= BATCH_SIZE:
                for bid, n, s, p in buffer:
                    grp = f.create_group(bid)
                    grp.create_dataset("note_encode", data=n, compression="lzf")
                    grp.create_dataset("sylphone_encode", data=s, compression="lzf")
                    grp.create_dataset("sylphones", data=p)
                buffer.clear()

        # flush remaining
        for bid, n, s, p in buffer:
            grp = f.create_group(bid)
            grp.create_dataset("note_encode", data=n)
            grp.create_dataset("sylphone_encode", data=s)
            grp.create_dataset("sylphones", data=p)

    print(f"Total features extracted',{feature_num}")

if __name__ == "__main__":
    main()