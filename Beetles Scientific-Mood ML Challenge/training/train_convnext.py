import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import json
import numpy as np
from tqdm import tqdm
import timm
import gc
from datasets import load_from_disk, concatenate_datasets
from sklearn.model_selection import KFold
from PIL import Image

# ===========================
# 1. 成功提交版的 CONFIG
# ===========================
CONFIG = {
    'dataset_path': './sentinel_beetles_local', 
    'save_dir': './weights_convnext',    
    'model_name': 'convnext_base.fb_in22k_ft_in1k', 
    'img_size': 224,                            
    'batch_size': 64,                           
    'epochs': 50,                               
    'lr': 1e-5,                                 
    'k_folds': 5,                               
    'num_workers': 4,
    'mapping_path': './mappings.json'
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 2. MODEL (與 model_old.py 100% 同步)
# ===========================
class ConvNext_HybridRegressor(nn.Module):
    def __init__(self, vision_dim=1024, num_species=152, num_domains=10, 
                 species_emb_dim=64, domain_emb_dim=16, hidden_size=512, num_outputs=6):
        super().__init__()
        # 訓練時 pretrained 必須設為 True 才能載入 ImageNet-22k 知識
        self.backbone = timm.create_model(CONFIG['model_name'], pretrained=True, num_classes=0)
        
        self.species_embedding = nn.Embedding(num_species, species_emb_dim, padding_idx=0)
        self.domain_embedding = nn.Embedding(num_domains, domain_emb_dim, padding_idx=0)
        total_input_dim = vision_dim + species_emb_dim + domain_emb_dim
        
        # V57.0 的 Head 結構
        self.regressor = nn.Sequential(
            nn.Linear(total_input_dim, hidden_size),
            nn.LayerNorm(hidden_size), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_outputs),
        )

    def forward_image(self, x):
        return F.normalize(self.backbone(x), dim=-1)

    def forward(self, images, species_idx, domain_idx):
        img_feat = self.forward_image(images)
        s_emb = self.species_embedding(species_idx)
        d_emb = self.domain_embedding(domain_idx)
        return self.regressor(torch.cat([img_feat, s_emb, d_emb], dim=1))

# ===========================
# 3. DATASET (HF 直讀與防呆)
# ===========================
class HFBeetleDataset(Dataset):
    def __init__(self, hf_dataset, s_map, d_map, transform=None):
        self.data = hf_dataset
        self.transform = transform
        self.s_map, self.d_map = s_map, d_map
        self.default_img = Image.new('RGB', (CONFIG['img_size'], CONFIG['img_size']), (0, 0, 0))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            image = item.get('image', None)
            if image is None or not isinstance(image, Image.Image):
                try: image = Image.open(item.get('file_path', '')).convert("RGB")
                except: image = self.default_img
            else: image = image.convert("RGB")
        except:
            image, item = self.default_img, {}

        if self.transform: image = self.transform(image)

        s_idx = self.s_map.get(item.get('scientificName', ''), 0)
        d_idx = self.d_map.get(str(item.get('domainID', '')), 0)
        
        try: 
            t = torch.tensor([item.get('SPEI_30d',0), item.get('SPEI_1y',0), item.get('SPEI_2y',0)], dtype=torch.float32)
        except: 
            t = torch.zeros(3, dtype=torch.float32)
        
        return image, s_idx, d_idx, t

# ===========================
# 4. TRAINING LOOP
# ===========================
def train_fold(fold, train_hf, val_hf, s_map, d_map):
    print(f"\n🚀 Fold {fold} | Model: ConvNeXt Base | Size: 224 | Batch: {CONFIG['batch_size']}")
    
    tf_train = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tf_val = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    loader_train = DataLoader(HFBeetleDataset(train_hf, s_map, d_map, tf_train), batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    loader_val = DataLoader(HFBeetleDataset(val_hf, s_map, d_map, tf_val), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

    model = ConvNext_HybridRegressor(num_species=len(s_map), num_domains=len(d_map)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # 標準的 MSE Loss
    criterion = nn.MSELoss(reduction='none') 

    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        loop = tqdm(loader_train, desc=f"Ep {epoch+1}", leave=False)
        for img, s, d, t in loop:
            img, s, d, t = img.to(device), s.to(device), d.to(device), t.to(device)
            optimizer.zero_grad()
            out = model(img, s, d)
            
            # 使用我們測試出能提升 R2 的加權 Loss
            mu = out[:, :3]
            log_var = out[:, 3:]
            log_var = torch.clamp(log_var, min=-5.0, max=5.0)
            precision = torch.exp(-log_var)
            raw_loss = 0.5 * precision * (mu - t)**2 + 0.5 * log_var
            weight = 1.0 + torch.abs(t) 
            loss = (raw_loss * weight).mean()
            
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, s, d, t in loader_val:
                img, s, d, t = img.to(device), s.to(device), d.to(device), t.to(device)
                val_loss += F.mse_loss(model(img, s, d)[:, :3], t).item()
        
        avg_val = val_loss / len(loader_val)
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), f"{CONFIG['save_dir']}/model_convnext_fold_{fold}.pth")
            
    print(f"✅ Fold {fold} Best MSE: {best_loss:.4f}")
    del model, optimizer; torch.cuda.empty_cache(); gc.collect()

def main():
    try:
        with open(CONFIG['mapping_path'], "r") as f: m = json.load(f)
    except:
        m = {'species_to_idx':{}, 'domain_to_idx':{}}
        
    print("📥 Loading dataset from disk...")
    ds = load_from_disk(CONFIG['dataset_path'])
    if isinstance(ds, dict) or "DatasetDict" in str(type(ds)):
        ds = concatenate_datasets([ds[k] for k in ds.keys()])
        
    kfold = KFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
    for fold, (t_idx, v_idx) in enumerate(kfold.split(np.arange(len(ds)))):
        train_fold(fold, ds.select(t_idx), ds.select(v_idx), m.get('species_to_idx',{}), m.get('domain_to_idx',{}))

if __name__ == '__main__': 
    main()
