import os

# ========= FILE PATH ========= #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data") 
 
IMAGE_DIR = os.path.join(DATA_DIR, "images")

METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")

RESULT_DIR = os.path.join(BASE_DIR, "results") 

DROP_COLUMN = [
    'patient_id',
    'lesion_id',
    'smoke',
    'drink',
    'background_father',
    'background_mother',
    'pesticide',
    'gender',
    'skin_cancer_history',
    'cancer_history',
    'has_piped_water',
    'has_sewage_system',
    'fitspatrick',
    'diameter_1',
    'diameter_2'
]

MAPPING_COL = [
    'itch',
    'grew',
    'hurt',
    'changed',
    'bleed',
    'elevation',
    'biopsed'
]

META_FEATURE_COLS = [
    'itch',
    'grew',
    'hurt',
    'changed',
    'bleed',
    'elevation',
    'biopsed',
    'region'
]

REGION_MAP = {
    'ARM': 0,
    'NECK': 1,
    'FACE': 2,
    'HAND': 3,
    'FOREARM': 4,
    'CHEST': 5,
    'NOSE': 6,
    'THIGH': 7,
    'SCALP': 8,
    'EAR': 9,
    'BACK': 10,
    'FOOT': 11,
    'ABDOMEN': 12,
    'LIP': 13,
    'UNKNOWN': -1,
}

BINARY_MAP = {
    'TRUE': 1,
    'FALSE': 0,
    'UNK': -1,
}

LABEL_MAP = {
    "BCC": 0,
    "SCC": 1,
    "MEL": 2,
    "SEK": 3,
    "ACK": 4,
    "NEV": 5,
}

IMAGE_SIZE = (224, 224) # b0 resNet 224,224  b1 240 240

LABEL_COL = "diagnostic"

RANDOM_SEED = 42

TEST_SPLIT = 0.2

NUM_CLASS = len(LABEL_MAP) # = 6

AUG_CONFIG = [0, 0, 0, 0, 0, 0] # 無
# AUG_CONFIG = [0, 4, 12, 3, 0, 3] # 有

BATCH_SIZE = 32

RF_N_ESTIMATOR = 100 

RF_MAX_DEPTH = None
# RF_MAX_DEPTH = 5
# RF_MAX_DEPTH = 10
# RF_MAX_DEPTH = 15

