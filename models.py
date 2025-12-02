import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights





class CNN_model(nn.Module):
    
    def __init__(self):
        super(CNN_model, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        base_model = efficientnet_b0(weights=weights)
        self.features = base_model.features  

    
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

                f_img = self.features(imgs)           # (B, 1280, 1, 1)
                f_img = torch.mean(f_img, dim=[2,3])  # (B, 1280)

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





    
      
