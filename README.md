# Skin Disease Classification using CNN & Random Forest

## 專案簡介 (Project Overview)
本專案為機器學習概論與實作期末專題報告。我們提出了一種 **Two-stage pipeline** 的方法來辨識皮膚疾病影像。

本研究結合了 **CNN (EfficientNet-B0)** 強大的特徵提取能力與 **Random Forest** 的分類能力，期望達到更客觀且快速的輔助診斷 。

## 研究方法 (Methodology)
我們的流程包含幾個主要階段：
1. **Load Data**: 讀取資料集
2. **Data Preprocessing**: 資料預處理，統一影像大小，並針對較少樣本數的類別做 data augmentation。
3. **Feature Extraction**: 使用預訓練的 EfficientNet-B0 從影像中提取特徵向量 (Feature Vector)。
4. **Classification**: 將影像特徵與病患 Metadata (如年齡、性別、部位) 串接 (Concatenation) 後，送入 Random Forest 進行最終分類。
5. **Evaluation metric**: 計算並呈現模型評估指標如 recall, precision, F2-score, 和 confusion matrix等

## 專案檔案結構 (Project Structure)

本專案包含以下主要程式檔案：

| 檔案名稱 | 說明 |
| :--- | :--- |
| `main.py` | **主程式**。控制完整研究流程：載入資料 -> 提取特徵 -> 訓練 RF 模型 -> 輸出評估結果。 |
| `preprocess.py` |**資料前處理**。包含影像的縮放 (Resize) 與補白 (Padding) 至 224x224 ，以及 Data Augmentation (旋轉) 和 Metadata 的編碼處理。 |
| `models.py` | **模型定義**。定義 EfficientNet-B0 特徵提取器與 Random Forest 分類器的架構。 |
| `dataset.py` | **資料集載入器**。負責讀取資料集影像與對應的 Metadata CSV 檔。 |
| `utils.py` | **評估工具**。包含計算 Accuracy, Precision, Recall 以及本專案重視的 **F2-score** 。 |
| `config.py` | **參數設定**。存放全域變數 (如影像路徑、Batch Size、Random Seed)。 |

## 資料集 (Dataset)
Mahdavi, A. (2020). Skin cancer (PAD-UFES-20) [Data set]. Kaggle. 
https://www.kaggle.com/datasets/mahdavi1202/skin-cancer

本研究使用 Skin cancer (PAD-UFES-20) 資料集，包含約 2300 張皮膚影像。
* **影像預處理**: 統一 Resize 並 Padding 至 224x224 像素。
* **類別**: 包含 基底細胞癌(Basal Cell Carcinoma, BCC)、鱗狀細胞癌(Squamous Cell Carcinoma, SCC)、黑色素瘤(Melanoma, MEL)、光化性角化病(Actinic Keratosis, ACK)、脂溢性角化病(Seborrheic Keratosis, SEK)和黑斑(Nevus, NEV) 等6大類。

## 組員 (Members)
* 112550062 莊詔允
* 112550066 張予綸
* 112610057 張予誠
