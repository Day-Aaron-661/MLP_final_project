import os

# ========= FILE PATH ========= #

# project資料夾絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# data 資料夾絕對路徑
DATA_DIR = os.path.join(BASE_DIR, "data") 
 
# images 資料夾路徑
IMAGE_DIR = os.path.join(DATA_DIR, "images")

# metadata 路徑
METADATA_DIR = os.path.join(DATA_DIR, "metadata.csv")

# result 存放路徑
RESULT_DIR = os.path.join(BASE_DIR, "results") 

# ========= data 相關參數 ========= #

IMAGE_SIZE = (224, 224)

RANDOM_SEED = 42

TEST_SPLIT = 0.2

CLASS_MAP = {
    "BCC": 0,
    "SCC": 1,
    "MEL": 2,
    "SEK": 3,
    "ACK": 4,
    "NEV": 5,
}

NUM_CLASS = len(CLASS_MAP) # = 6

# ========= model 相關參數 ========= #

BATCH_SIZE = 32

RF_N_ESTIMATOR = 100 

RF_MAX_DEPTH = None
# RF_MAX_DEPTH = 10
# RF_MAX_DEPTH = 50
# RF_MAX_DEPTH = 100