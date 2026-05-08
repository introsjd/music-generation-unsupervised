import numpy as np


def rhythm_diversity(piano_roll, threshold=0.5):
    durations = []
    for pitch in range(128):
        count = 0
        for t in range(piano_roll.shape[0]):
            if piano_roll[t, pitch] > threshold:
                count += 1
            elif count > 0:
                durations.append(count)
                count = 0
    if len(durations) == 0:
        return 0.0
    return round(len(set(durations)) / len(durations), 3)


def repetition_ratio(piano_roll, window=4):
    patterns = []
    for t in range(piano_roll.shape[0] - window):
        pat = tuple(piano_roll[t:t+window].flatten().round(1))
        patterns.append(pat)
    if len(patterns) == 0:
        return 0.0
    return round(1 - len(set(patterns)) / len(patterns), 3)


def pitch_histogram(piano_roll, threshold=0.5):
    hist = np.zeros(12)
    for t in range(piano_roll.shape[0]):
        for pitch in range(128):
            if piano_roll[t, pitch] > threshold:
                hist[pitch % 12] += 1
    total = hist.sum()
    return hist / total if total > 0 else hist


def histogram_similarity(roll1, roll2):
    p = pitch_histogram(roll1)
    q = pitch_histogram(roll2)
    return round(float(1 - np.sum(np.abs(p - q)) / 2), 3)
