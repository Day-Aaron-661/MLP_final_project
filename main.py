import config
import dataset
import preprocess
import models
import utils
import numpy as np
from torch.utils.data import DataLoader

# 假設使用 PyTorch 或 TensorFlow，這裡以通用的命名為主

def main():

    # --- 定義動態增強 (Augmentation) ---
    # 這是要在 __getitem__ 裡隨機做的
    train_aug = preprocess.get_augmentation() # 回傳隨機旋轉函數
    
    print("正在處理訓練集 (含 Data Augmentation)...")
    print("=== 初始化訓練集 ===")
    train_dataset = dataset.SkinLesionDataset(
        csv_file=METADATA_DIR,
        img_dir=IMAGE_DIR,
        mode='train',
        augmentation=train_aug # 把增強邏輯傳進去
    )
    
    print("=== 初始化測試集 ===")
    test_dataset = dataset.SkinLesionDataset(
        csv_file=Test_csv_path,
        img_dir=IMAGE_DIR,
        mode='test',
        augmentation=None # 測試集不做增強
    )

    print("=== 封裝 Data Loader ===")
    # --- 建立 DataLoader ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("=== CNN 特徵提取 (Feature Extraction) ===")
    # 這裡的邏輯與之前類似，但輸入的是已經處理好的 loader
    cnn_model = models.CNN_model()
    
    print("提取訓練集特徵...")
    X_train_features = cnn_model.extract_features(cnn_model, train_loader)
    # 此時 X_train_features 已經是 (CNN特徵 + Metadata) 的混合向量 [cite: 68, 110]
    
    print("提取測試集特徵...")
    X_test_features = cnn_model.extract_features(cnn_model, test_loader)

    print("=== Random Forest 分類 (Classification) ===")
    rf_model = models.RF_classifier()
    
    print("訓練 RF 模型...")
    rf_model.train(X_train_features, train_labels) # [cite: 69]
    
    print("=== 評估與結果 (Evaluation) ===")
    predictions = rf_model.predict(X_test_features)
    
    # 計算 Recall, Precision, F2-score [cite: 71-78]
    metrics = utils.calculate_metrics(test_labels, predictions)
    utils.print_evaluation_report(metrics)

if __name__ == "__main__":
    main()
