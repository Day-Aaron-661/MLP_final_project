# train_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config as cfg
import dataset
import preprocess
from models import FineTuneModel

def train_cnn_stage():

    all_meta = dataset.load_csv( csv_path=cfg.METADATA_PATH )
    all_clean_meta = preprocess.get_clean_data( all_meta )
    
    train_meta_df, test_meta_df = dataset.metadata_split( all_clean_meta, cfg.TEST_SPLIT, cfg.RANDOM_SEED )

    train_imgs = dataset.Skin_Datasest(
        metadata_df=train_meta_df,
        img_dir=cfg.IMAGE_DIR,
        mode='train',
        augmentation=cfg.AUG_CONFIG
    )
    train_loader = DataLoader(train_imgs, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FineTuneModel(num_classes=cfg.NUM_CLASS).to(device)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 15
    print(f"開始微調 CNN (共 {epochs} Epochs)...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, meta, labels in train_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

    # save_path = "fine_tuned_efficientnetb0.pth"
    # save_path = "fine_tuned_efficientnetb1.pth"
    save_path = "fine_tuned_efficientnetresnet.pth"
    torch.save(model.base_model.state_dict(), save_path)
    print(f"模型權重已儲存至: {save_path}")

if __name__ == "__main__":
    train_cnn_stage()

