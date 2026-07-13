import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

HUMBUGDB_AUDIO_DIR = os.path.join(DATA_DIR, "audio")
HUMBUGDB_METADATA_PATH = os.path.join(DATA_DIR, "metadata", "neurips_2021_zenodo_0_0_1.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SAMPLE_RATE = 16000
DURATION = None

N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64
FMIN = 20
FMAX = 8000

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

NUM_CLASSES = 2