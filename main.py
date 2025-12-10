import config as cfg
import dataset
import preprocess
import models
import utils
import numpy as np
from torch.utils.data import DataLoader


def main():

    # 先載入 metadata，然後去除一些 nan 和不要的欄位
    all_meta = dataset.load_csv( csv_path=cfg.METADATA_PATH )
    all_clean_meta = preprocess.get_clean_data( all_meta )
    
    # 將 metadata 分成 train 和 test
    train_meta_df, test_meta_df = dataset.metadata_split( all_clean_meta, cfg.TEST_SPLIT, cfg.RANDOM_SEED )

    print("正在處理訓練集 (含 Data Augmentation)...")
    print("=== 初始化訓練集 ===")
    train_imgs = dataset.Skin_Datasest(
        metadata_df=train_meta_df,
        img_dir=cfg.IMAGE_DIR,
        mode='train',
        augmentation=cfg.AUG_CONFIG # 把增強邏輯傳進去
    )
    
    print("=== 初始化測試集 ===")
    test_imgs = dataset.Skin_Datasest(
        metadata_df=test_meta_df,
        img_dir=cfg.IMAGE_DIR,
        mode='test',
        augmentation=None # 測試集不做增強
    )

    print("=== 封裝 Data Loader ===")
    # --- 建立 DataLoader ---
    train_loader = DataLoader(train_imgs, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_imgs, batch_size=cfg.BATCH_SIZE, shuffle=False)

    cnn_classifier = models.CNN_classifier(
        num_classes=cfg.NUM_CLASS,
        learning_rate=1e-4, 
        device="cuda"
    )

    print("=== CNN 特徵提取 (Feature Extraction) ===")
    # 這裡的邏輯與之前類似，但輸入的是已經處理好的 loader
    cnn_model = models.CNN_model(weight_path="fine_tuned_efficientnet.pth")
    
    print("提取訓練集特徵...")
    X_train_features, train_labels = cnn_model.extract_features(cnn_model, train_loader)
    # 此時 X_train_features 已經是 (CNN特徵 + Metadata) 的混合向量
    
    print("提取測試集特徵...")
    X_test_features, test_labels = cnn_model.extract_features(cnn_model, test_loader)

    print("=== Random Forest 分類 (Classification) ===")
    rf_model = models.RF_classifier()
    
    print("訓練 RF 模型...")
    # rf_model.train(X_train_features, train_labels)
    best_rf_model = models.tune_rf_hyperparameters(X_train_features, train_labels)

    # 直接用這個 best_rf_model 預測
    print("=== 評估與結果 (Evaluation) ===")
    # predictions = rf_model.predict(X_test_features)
    predictions = best_rf_model.predict(X_test_features)
    
    # 計算 Confusion Matrix, Accuracy, Precision, Recall, F2-score
    cm, classes = utils.confusion_matrix(test_labels, predictions)
    print("\n===== Evaluation Report =====")
    print("Confusion Matrix:")
    print(cm)
    print("\nClasses:", classes)
    print(f"\nAccuracy: {utils.accuracy(cm):.4f}")
    precision = utils.precision(cm)
    print(f"Precision: {precision:.4f}")
    recall = utils.recall(cm)
    print(f"Recall: {recall:.4f}")
    print(f"F2-score: {utils.f2_score(precision, recall):.4f}")

# def main():

#     # 先載入 metadata，然後去除一些 nan 和不要的欄位
#     all_meta = dataset.load_csv( csv_path=cfg.METADATA_PATH )
#     all_clean_meta = preprocess.get_clean_data( all_meta )
    
#     # 將 metadata 分成 train 和 test
#     train_meta_df, test_meta_df = dataset.metadata_split( all_clean_meta, cfg.TEST_SPLIT, cfg.RANDOM_SEED )

#     print("正在處理訓練集 (含 Data Augmentation)...")
#     print("=== 初始化訓練集 ===")
#     train_imgs = dataset.Skin_Datasest(
#         metadata_df=train_meta_df,
#         img_dir=cfg.IMAGE_DIR,
#         mode='train',
#         augmentation=cfg.AUG_CONFIG # 把增強邏輯傳進去
#     )
    
#     print("=== 初始化測試集 ===")
#     test_imgs = dataset.Skin_Datasest(
#         metadata_df=test_meta_df,
#         img_dir=cfg.IMAGE_DIR,
#         mode='test',
#         augmentation=None # 測試集不做增強
#     )

#     print("=== 封裝 Data Loader ===")
#     # --- 建立 DataLoader ---
#     train_loader = DataLoader(train_imgs, batch_size=cfg.BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_imgs, batch_size=cfg.BATCH_SIZE, shuffle=False)

#     cnn_classifier = models.CNN_classifier(
#         num_classes=cfg.NUM_CLASS,
#         learning_rate=1e-4, 
#         device="cuda"
#     )

#     cnn_classifier.train(train_loader, epochs=5)

#     # === 4. 預測與評估 ===
#     # 直接用 CNN 預測
#     predictions = cnn_classifier.predict(test_loader)
    
#     # 取得真實標籤 (從 dataset 中拿)
#     test_labels = test_imgs.labels # 或是你之前用來存 label 的 list
    
#     # 計算 Confusion Matrix, Accuracy, Precision, Recall, F2-score [cite: 71-78]
#     cm, classes = utils.confusion_matrix(test_labels, predictions)
#     print("\n===== Evaluation Report =====")
#     print("Confusion Matrix:")
#     print(cm)
#     print("\nClasses:", classes)
#     print(f"\nAccuracy: {utils.accuracy(cm):.4f}")
#     precision = utils.precision(cm)
#     print(f"Precision: {precision:.4f}")
#     recall = utils.recall(cm)
#     print(f"Recall: {recall:.4f}")
#     print(f"F2-score: {utils.f2_score(precision, recall):.4f}")

if __name__ == "__main__":
    main()

