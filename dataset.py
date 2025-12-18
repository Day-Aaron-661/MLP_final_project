import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import typing as t
import os
import cv2
import numpy as np
import config as cfg
import preprocess


class Skin_Datasest(Dataset):
    def __init__( self, metadata_df, img_dir, mode='test', augmentation=None ):
        self.mode = mode
        self.images = []
        self.labels = []
        self.metadata_vectors = []
        
        self.metadata_features_mat = metadata_df[cfg.META_FEATURE_COLS].values.astype(np.float32)
        self.meta_labels = metadata_df[cfg.LABEL_COL].values.astype(np.int32)
        self.img_ids = metadata_df['img_id'].values
        
        print(f"[{mode}] 開始載入與處理影像 (共 {len(metadata_df)} 筆原圖)...")
        
        for idx in range(len(metadata_df)):
            
            img_path = os.path.join(img_dir, self.img_ids[idx])
            # img = cv2.imread(img_path)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            base_img = preprocess.resize_and_padding(img, target_size=cfg.IMAGE_SIZE)
            
            self._add_sample(base_img, self.metadata_features_mat[idx], self.meta_labels[idx])
            
            if augmentation:
                num_augments = augmentation[self.meta_labels[idx]]
                
                for i in range(num_augments):
                    aug_img = preprocess.augment_image(base_img)
                    
                    self._add_sample(aug_img,  self.metadata_features_mat[idx], self.meta_labels[idx])

        print(f"[{mode}] 載入完成。最終資料總數: {len(self.images)} (含增強)")

    def _add_sample( self, image, meta, label ):
        image = image.astype('float32') / 255.0
        image = torch.tensor(image).permute(2, 0, 1)
        
        self.images.append(image)
        self.metadata_vectors.append(meta)
        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        return self.images[idx], torch.tensor(self.metadata_vectors[idx]).float(), torch.tensor(self.labels[idx]).int()


def metadata_split( all_meta: pd.DataFrame, test_split=cfg.TEST_SPLIT, random_seed=cfg.RANDOM_SEED ) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    if all_meta.shape[0] == 0:
        print("no metadata")
        return
    
    X = all_meta
    y = all_meta[cfg.LABEL_COL]

    train_meta_df, test_meta_df = train_test_split(
        X,
        test_size=test_split,
        stratify=y,
        random_state=random_seed
    )

    return train_meta_df, test_meta_df


def load_csv( csv_path=cfg.METADATA_PATH ):
    metadata = pd.read_csv(csv_path)
    return metadata


