import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier




class CNN_model(nn.Module):
    
    def __init__(self):
        super(CNN_model, self).__init__()
      
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # H/2, W/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # H/4, W/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))   # (128, 1, 1)
        )
    
    def extract_features(self, model, dataloader, device="cuda"):
        model = model.to(device)
        model.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                imgs, meta, labels = batch
                imgs = imgs.to(device)
                meta = meta.to(device)

                f_img = self.features(imgs)      # (B, 128, 1, 1)
                f_img = torch.flatten(f_img, 1)  # (B, 128)

                f_combined = torch.cat([f_img, meta], dim=1)

                all_features.append(f_combined.cpu().numpy())
                all_labels.append(labels.numpy())

        # concatenate
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        print(f"[CNN] extract_features 完成，shape = {all_features.shape}")
        return all_features, all_labels




class RF_classifier:
    def __init__(self, n_estimators=300, random_state=666):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

    # 訓練------------------------------------------------
    def train(self, X_train, y_train):
        print("[RF] 開始訓練 RF...")
        self.model.fit(X_train, y_train)
        print("[RF] 訓練完成")

    # 預測------------------------------------------------
    def predict(self, X_test):
        print("[RF] 開始預測...")
        return self.model.predict(X_test)





    
      
