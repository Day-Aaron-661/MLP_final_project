import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models import resnet50, ResNet50_Weights

class FineTuneModel(nn.Module):
    def __init__(self, num_classes=6):
        super(FineTuneModel, self).__init__()

        # weights = EfficientNet_B0_Weights.DEFAULT
        # self.base_model = efficientnet_b0(weights=weights)
        # weights = EfficientNet_B1_Weights.DEFAULT
        # self.base_model = efficientnet_b1(weights=weights)
        # self.base_model.classifier = nn.Sequential(
        #     nn.Dropout(p=0.2, inplace=True),
        #     nn.Linear(1280, num_classes),
        # )
        
        # ResNet50
        weights = ResNet50_Weights.DEFAULT
        self.base_model = resnet50(weights=weights)
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, num_classes) 
        )

    def forward(self, x):
        return self.base_model(x)


class CNN_model(nn.Module):
    
    def __init__(self, weight_path=None):
        super(CNN_model, self).__init__()

        # efficientNet
        # base_model = efficientnet_b0(weights=None)
        # base_model = efficientnet_b1(weights=None)
        # base_model.classifier = nn.Sequential(
        #     nn.Dropout(p=0.2, inplace=True),
        #     nn.Linear(1280, 6),
        # )

        # # resNet
        base_model = resnet50(weights=None)
        base_model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 6)
        )
        
        if weight_path:
            print(f"正在載入微調權重: {weight_path}")
            state_dict = torch.load(weight_path)
            base_model.load_state_dict(state_dict)
        else:
            print("使用 ImageNet 預設權重")
            # base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT) # b0
            # base_model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT) # b1
            base_model = resnet50(weights=ResNet50_Weights.DEFAULT) # resNet

        # self.features = base_model.features # b0, b1
        self.features = nn.Sequential(*list(base_model.children())[:-1]) # resNet

    
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

                f_img = self.features(imgs)
                f_img = torch.mean(f_img, dim=[2,3])

                f_combined = torch.cat([f_img, meta], dim=1)

                all_features.append(f_combined.cpu().numpy())
                all_labels.append(labels.numpy())

        # concatenate
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        print(f"[CNN] extract_features 完成，shape = {all_features.shape}")
        return all_features, all_labels



class RF_classifier:
    def __init__(self, n_estimators=20, random_state=666):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
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


from sklearn.model_selection import RandomizedSearchCV


def tune_rf_hyperparameters(X_train, y_train):
    print("正在進行 Random Forest 超參數調優 (避免 Overfit)...")
    
    # 1. 定義參數網格
    # 這裡的範圍是專門為了「抗 Overfit」設計的
    param_dist = {
        'n_estimators': [200, 300, 500],        # 樹越多通常越穩定
        'max_depth': [10, 15, 20, 30],         # 限制深度！
        'min_samples_split': [5, 10, 20],            # 避免切分太細
        'min_samples_leaf': [2, 4, 8, 12],           # 關鍵：葉子不能只有 1 個樣本
        'max_features': ['sqrt', 'log2'],            # 限制每棵樹看到的特徵
        'bootstrap': [True],                         # 必須為 True 才有 Bagging 效果
        'class_weight': ['balanced']                 # 記得保留這個解決不平衡
    }

    # 2. 初始化 RF
    rf = RandomForestClassifier(random_state=42)

    # 3. 初始化隨機搜尋 (比 GridSearch 快很多)
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=50,             # 隨機嘗試 50 種組合
        cv=5,                  # 5-Fold 交叉驗證 (最穩健的評估)
        verbose=0,
        random_state=42,
        n_jobs=-1,             # 用盡所有 CPU 核心
        scoring='f1_macro'     # 針對多類別不平衡，優化 F1-score 
    )

    # 4. 開始訓練搜尋
    rf_random.fit(X_train, y_train)

    print(f"最佳參數組合: {rf_random.best_params_}")
    print(f"最佳驗證分數: {rf_random.best_score_:.4f}")
    
    return rf_random.best_estimator_ # 回傳調校好的最佳模型


# models.py 新增部分

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from tqdm import tqdm # 記得 pip install tqdm 來看進度條

class CNN_classifier:
    def __init__(self, num_classes=6, learning_rate=1e-4, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # self.model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )

        # self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(2048, num_classes)
        # )
        
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader, epochs=15):
        """
        取代原本 RF.train 的功能，這裡直接進行 CNN 的訓練
        """
        print(f"[CNN Classifier] 開始訓練 (Epochs: {epochs})...")
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for images, meta, labels in loop:

                images = images.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)
                
                # Forward
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward
                loss.backward()
                self.optimizer.step()
                
                # 統計
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 更新進度條資訊
                loop.set_postfix(loss=loss.item(), acc=correct/total)
                
        print("[CNN Classifier] 訓練完成")

    def predict(self, test_loader):
        """
        取代原本 RF.predict 的功能
        回傳 numpy array 格式的預測結果，讓 utils.calculate_metrics 可以直接用
        """
        print("[CNN Classifier] 正在進行預測...")
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for images, meta, labels in tqdm(test_loader, desc="Predicting"):
                images = images.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                
        return all_preds
