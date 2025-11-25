# Skin Disease Classification using CNN & Random Forest

## 專案簡介 (Project Overview)
[cite_start]本專案為機器學習概論與實作期中專題報告。我們提出了一種 **Two-stage pipeline** 的方法來辨識皮膚疾病影像 [cite: 65]。

[cite_start]傳統 CNN 模型雖然強大，但存在「黑箱問題」且訓練成本高 [cite: 26, 27][cite_start]。本研究結合了 **CNN (EfficientNet-B0)** 強大的特徵提取能力與 **Random Forest** 的分類能力，期望達到更客觀且快速的輔助診斷 [cite: 9, 220]。

## 研究方法 (Methodology)
[cite_start]我們的流程包含兩個主要階段 [cite: 65, 193]：
1.  [cite_start]**Feature Extraction**: 使用預訓練的 EfficientNet-B0 [cite: 82] 從影像中提取特徵向量 (Feature Vector)。
2.  [cite_start]**Classification**: 將影像特徵與病患 Metadata (如年齡、性別、部位) 串接 (Concatenation) 後 [cite: 110]，送入 Random Forest 進行最終分類。

## 專案檔案結構 (Project Structure)

本專案包含以下主要程式檔案：

| 檔案名稱 | 說明 |
| :--- | :--- |
| `main.py` | **主程式**。控制完整研究流程：載入資料 -> 提取特徵 -> 訓練 RF 模型 -> 輸出評估結果。 |
| `preprocess.py` | [cite_start]**資料前處理**。包含影像的縮放 (Resize) 與補白 (Padding) 至 224x224 [cite: 97, 98]，以及 Data Augmentation (旋轉) 和 Metadata 的編碼處理。 |
| `models.py` | **模型定義**。定義 EfficientNet-B0 特徵提取器與 Random Forest 分類器的架構。 |
| `dataset.py` | **資料集載入器**。負責讀取 HAM10000 影像路徑與對應的 Metadata CSV 檔。 |
| `utils.py` | [cite_start]**評估工具**。包含計算 Accuracy, Precision, Recall 以及本專案重視的 **F2-score** [cite: 78]。 |
| `config.py` | **參數設定**。存放全域變數 (如影像路徑、Batch Size、Random Seed)。 |

## 資料集 (Dataset)
[cite_start]本研究使用 **HAM10000** 資料集 [cite: 28]，包含約 10,015 張皮膚鏡影像。
* [cite_start]**影像預處理**: 統一 Resize 並 Padding 至 224x224 像素 [cite: 98]。
* [cite_start]**類別**: 包含 akiec, bcc, bkl, df, mel, nv, vasc 等 7 大類 [cite: 31]。

## 環境需求 (Requirements)
* Python 3.x
* PyTorch / TensorFlow (視實作而定)
* scikit-learn
* pandas, numpy
* Pillow (PIL)

## 協作者 (Contributors)
* 112550062 莊詔允
* 112550066 張予綸
* 112610057 張予誠
[cite_start][cite: 2]
