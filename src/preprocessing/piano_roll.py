import numpy as np
import pretty_midi


def piano_roll_to_midi(piano_roll, fs=4, threshold=0.3, tempo=120):
    midi  = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)
    seconds_per_step = 60.0 / (tempo * fs)
    for pitch in range(128):
        note_on    = False
        start_time = 0.0
        for t in range(piano_roll.shape[0]):
            active = piano_roll[t, pitch] > threshold
            if active and not note_on:
                start_time = t * seconds_per_step
                note_on    = True
            elif not active and note_on:
                end_time = t * seconds_per_step
                if end_time > start_time:
                    piano.notes.append(pretty_midi.Note(
                        velocity=80, pitch=pitch,
                        start=start_time, end=end_time))
                note_on = False
    midi.instruments.append(piano)
    return midi


def piano_roll_to_tokens(piano_roll_step, threshold=0.5):
    active = np.where(piano_roll_step > threshold)[0]
    if len(active) == 0:
        return 128
    return int(active[np.argmax(piano_roll_step[active])])


def tokens_to_piano_roll(tokens):
    roll = np.zeros((len(tokens), 128), dtype=np.float32)
    for t, tok in enumerate(tokens):
        if tok < 128:
            roll[t, tok] = 1.0
    return roll
