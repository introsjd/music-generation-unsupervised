import pretty_midi
import numpy as np
import os
from tqdm import tqdm


def midi_to_piano_roll(filepath, fs=4):
    try:
        midi = pretty_midi.PrettyMIDI(filepath)
        roll = midi.get_piano_roll(fs=fs)
        return (roll > 0).astype(np.float32)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def segment_piano_roll(piano_roll, segment_len=64, hop=32):
    segments = []
    T = piano_roll.shape[1]
    for start in range(0, T - segment_len + 1, hop):
        seg = piano_roll[:, start:start + segment_len]
        if seg.sum() > 10:
            segments.append(seg)
    return segments


def build_dataset(raw_midi_folder, segment_len=64,
                  hop=32, fs=4, max_files=150):
    midi_files = [
        f for f in os.listdir(raw_midi_folder)
        if f.endswith('.midi') or f.endswith('.mid')
    ][:max_files]
    print(f"Processing {len(midi_files)} MIDI files...")
    all_segments = []
    skipped = 0
    for fname in tqdm(midi_files, desc="Converting"):
        fpath = os.path.join(raw_midi_folder, fname)
        roll  = midi_to_piano_roll(fpath, fs=fs)
        if roll is None or roll.shape[1] < segment_len:
            skipped += 1
            continue
        all_segments.extend(segment_piano_roll(roll, segment_len, hop))
    print(f"Skipped: {skipped} | Total segments: {len(all_segments)}")
    return np.array(all_segments)
