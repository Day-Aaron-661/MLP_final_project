import os
import cv2
import numpy as np
import pandas as pd
import typing as t
import config as cfg
from sklearn.preprocessing import StandardScaler


def get_clean_data( meta: pd.DataFrame ):

    clean_meta = meta.drop( columns = cfg.DROP_COLUMN)
    clean_meta = clean_meta.dropna()

    for col in cfg.MAPPING_COL:
        clean_meta[col] = clean_meta[col].astype(str).str.upper().str.strip()
        clean_meta[col] = clean_meta[col].map( cfg.BINARY_MAP )

        clean_meta[col] = clean_meta[col].fillna(-1).astype(int)

    clean_meta['region'] = clean_meta['region'].astype(str).str.upper().str.strip()
    clean_meta['region'] = clean_meta['region'].map( cfg.REGION_MAP )
    clean_meta['region'] = clean_meta['region'].fillna(-1).astype(int)

    clean_meta['diagnostic'] = clean_meta['diagnostic'].astype(str).str.upper().str.strip()
    clean_meta['diagnostic'] = clean_meta['diagnostic'].map( cfg.LABEL_MAP )
    clean_meta['diagnostic'] = clean_meta['diagnostic'].fillna(-1).astype(int)

    return clean_meta


def resize_and_padding( img, target_size=cfg.IMAGE_SIZE ):

    img_h, img_w = img.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    resized_img = cv2.resize( img, (new_w, new_h), interpolation=cv2.INTER_LINEAR )

    padding_background = np.full( (target_h, target_w, 3) , 255, dtype=np.uint8 )

    x_offset = ( target_w - new_w ) // 2
    y_offset = ( target_h - new_h ) // 2

    padding_background[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_img

    return padding_background


def augment_image( image, rotation_range: int = 120 ):

    h, w = image.shape[:2]

    angle = np.random.uniform(-rotation_range, rotation_range)
    
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    
    rotated = cv2.warpAffine(
        image, 
        M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255)
    )


    return rotated
