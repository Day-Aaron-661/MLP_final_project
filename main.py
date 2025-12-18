import config as cfg
import dataset
import preprocess
import models
import utils
import numpy as np
from torch.utils.data import DataLoader


def main():

    all_meta = dataset.load_csv( csv_path=cfg.METADATA_PATH )
    all_clean_meta = preprocess.get_clean_data( all_meta )

    train_meta_df, test_meta_df = dataset.metadata_split( all_clean_meta, cfg.TEST_SPLIT, cfg.RANDOM_SEED )

    print("正在處理訓練集 (含 Data Augmentation)...")
    print("=== 初始化訓練集 ===")
    train_imgs = dataset.Skin_Datasest(
        metadata_df=train_meta_df,
        img_dir=cfg.IMAGE_DIR,
        mode='train',
        augmentation=cfg.AUG_CONFIG
    )
    
    print("=== 初始化測試集 ===")
    test_imgs = dataset.Skin_Datasest(
        metadata_df=test_meta_df,
        img_dir=cfg.IMAGE_DIR,
        mode='test',
        augmentation=cfg.AUG_CONFIG
    )

    print("=== 封裝 Data Loader ===")
    train_loader = DataLoader(train_imgs, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_imgs, batch_size=cfg.BATCH_SIZE, shuffle=False)

    print("=== CNN 特徵提取 (Feature Extraction) ===")
    cnn_model = models.CNN_model(weight_path="fine_tuned_efficientnetb0.pth")
    # cnn_model = models.CNN_model(weight_path="fine_tuned_efficientnetb1.pth")
    # cnn_model = models.CNN_model(weight_path="fine_tuned_efficientnetresnet.pth")
    
    print("提取訓練集特徵...")
    X_train_features, train_labels = cnn_model.extract_features(cnn_model, train_loader)
    
    print("提取測試集特徵...")
    X_test_features, test_labels = cnn_model.extract_features(cnn_model, test_loader)

    print("=== Random Forest 分類 (Classification) ===")
    rf_model = models.RF_classifier()
    
    print("訓練 RF 模型...")
    rf_model.train(X_train_features, train_labels)

    print("=== 評估與結果 (Evaluation) ===")
    predictions = rf_model.predict(X_test_features)
    # predictions = best_rf_model.predict(X_test_features)
    
    cm, classes = utils.confusion_matrix(test_labels, predictions)
    print("\n===== Evaluation Report =====")
    print("Confusion Matrix:")
    print(cm)
    utils.plot_confusion_matrix(cm, classes)
    print("\nClasses:", classes)
    print(f"\nAccuracy: {utils.accuracy(cm):.4f}")
    precision = utils.precision(cm)
    print(f"Precision: {precision:.4f}")
    recall = utils.recall(cm)
    print(f"Recall: {recall:.4f}")
    print(f"F2-score: {utils.f2_score(precision, recall):.4f}")

# def main():

#     all_meta = dataset.load_csv( csv_path=cfg.METADATA_PATH )
#     all_clean_meta = preprocess.get_clean_data( all_meta )
    
#     train_meta_df, test_meta_df = dataset.metadata_split( all_clean_meta, cfg.TEST_SPLIT, cfg.RANDOM_SEED )

#     print("正在處理訓練集 (含 Data Augmentation)...")
#     print("=== 初始化訓練集 ===")
#     train_imgs = dataset.Skin_Datasest(
#         metadata_df=train_meta_df,
#         img_dir=cfg.IMAGE_DIR,
#         mode='train',
#         augmentation=cfg.AUG_CONFIG
#     )
    
#     print("=== 初始化測試集 ===")
#     test_imgs = dataset.Skin_Datasest(
#         metadata_df=test_meta_df,
#         img_dir=cfg.IMAGE_DIR,
#         mode='test',
#         augmentation=cfg.AUG_CONFIG
#     )

#     print("=== 封裝 Data Loader ===")
#     train_loader = DataLoader(train_imgs, batch_size=cfg.BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_imgs, batch_size=cfg.BATCH_SIZE, shuffle=False)

#     cnn_classifier = models.CNN_classifier(
#         num_classes=cfg.NUM_CLASS,
#         learning_rate=1e-4, 
#         device="cuda"
#     )

#     cnn_classifier.train(train_loader, epochs=5)

#     predictions = cnn_classifier.predict(test_loader)

#     test_labels = test_imgs.labels
    
#     cm, classes = utils.confusion_matrix(test_labels, predictions)
#     print("\n===== Evaluation Report =====")
#     print("Confusion Matrix:")
#     print(cm)
#     utils.plot_confusion_matrix(cm, classes)
#     print("\nClasses:", classes)
#     print(f"\nAccuracy: {utils.accuracy(cm):.4f}")
#     precision = utils.precision(cm)
#     print(f"Precision: {precision:.4f}")
#     recall = utils.recall(cm)
#     print(f"Recall: {recall:.4f}")
#     print(f"F2-score: {utils.f2_score(precision, recall):.4f}")

if __name__ == "__main__":
    main()



