# Project-wide configuration
BASE_DIR      = "/content/drive/MyDrive/CSE425_Music_Generation"
RAW_MIDI_DIR  = BASE_DIR + "/data/raw_midi"
PROCESSED_DIR = BASE_DIR + "/data/processed"
MODELS_DIR    = BASE_DIR + "/models"
OUTPUTS_DIR   = BASE_DIR + "/outputs"
PLOTS_DIR     = BASE_DIR + "/outputs/plots"

SEGMENT_LEN = 64
HOP         = 32
FS          = 4
MAX_FILES   = 150

BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-3
LATENT_DIM  = 64
HIDDEN_SIZE = 256
VOCAB_SIZE  = 129
BETA        = 1.0
