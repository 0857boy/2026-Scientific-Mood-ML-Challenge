import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import json
import numpy as np
import math
from tqdm import tqdm
import open_clip
import gc
from datasets import load_from_disk, concatenate_datasets
from sklearn.model_selection import KFold
from PIL import Image


CONFIG = {
    'dataset_path': './sentinel_beetles_local',
    'save_dir': './weights_bioclip_golden',
    'img_size': 224,                  # 依照您的需求改為 224, 336, 或 448
    'batch_size': 32,                 # 224: 32 | 336: 16 | 448: 8
    'epochs': 15,                     
    'lr_head': 1e-4,                
    'lr_backbone': 1e-5,              
    'unfreeze_layers': 2,            
    'k_folds': 5,
    'num_workers': 4,
    'mapping_path': './mappings.json'
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# 2. MODEL (BioClip 黃金版)
# ===========================
class BioClip2_HybridRegressor(nn.Module):
    def __init__(self, bioclip, num_species=152, num_domains=10, 
                 vision_dim=768, hidden_size=512, n_last_trainable_resblocks=2):
        super().__init__()
        self.bioclip = bioclip
        self.species_embedding = nn.Embedding(num_species, 64, padding_idx=0)
        self.domain_embedding = nn.Embedding(num_domains, 16, padding_idx=0)
        
        self.regressor = nn.Sequential(
            nn.Linear(vision_dim + 64 + 16, hidden_size),
            nn.LayerNorm(hidden_size), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 6),
        )

        # --- 凍結與解凍策略 ---
        # 1. 先全凍結
        for param in self.bioclip.parameters():
            param.requires_grad = False
            
        # 2. 解凍最後 N 個 ResBlocks
        if n_last_trainable_resblocks > 0:
            for r in self.bioclip.visual.transformer.resblocks[-n_last_trainable_resblocks:]:
                for param in r.parameters():
                    param.requires_grad = True

    def _resize_pos_embed(self, pos_embed, target_seq_len):
        if pos_embed.shape[0] == target_seq_len: return pos_embed
        cls_token = pos_embed[0:1, :]
        grid_tokens = pos_embed[1:, :]
        dim = grid_tokens.shape[1]
        old_grid_size = int(math.sqrt(grid_tokens.shape[0]))
        new_grid_size = int(math.sqrt(target_seq_len - 1))
        grid_tokens = grid_tokens.reshape(1, old_grid_size, old_grid_size, dim).permute(0, 3, 1, 2)
        new_grid_tokens = F.interpolate(grid_tokens, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
        new_grid_tokens = new_grid_tokens.permute(0, 2, 3, 1).reshape(-1, dim)
        return torch.cat((cls_token, new_grid_tokens), dim=0)

    def forward_image(self, x):
        x = self.bioclip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        _cls = self.bioclip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([_cls, x], dim=1)
        pos_embed = self.bioclip.visual.positional_embedding.to(x.dtype)
        if pos_embed.shape[0] != x.shape[1]: 
            pos_embed = self._resize_pos_embed(pos_embed, x.shape[1])
        x = x + pos_embed
        if hasattr(self.bioclip.visual, 'ln_pre'): x = self.bioclip.visual.ln_pre(x)
        if not self.bioclip.visual.transformer.batch_first: x = x.transpose(0, 1).contiguous()
        for r in self.bioclip.visual.transformer.resblocks: x = r(x, attn_mask=None)
        if not self.bioclip.visual.transformer.batch_first: x = x.transpose(0, 1)
        pooled, _ = self.bioclip.visual._pool(x)
        if self.bioclip.visual.proj is not None: pooled = pooled @ self.bioclip.visual.proj
        return F.normalize(pooled, dim=-1)

    def forward(self, images, species_idx, domain_idx):
        img_feat = self.forward_image(images)
        s_emb = self.species_embedding(species_idx)
        d_emb = self.domain_embedding(domain_idx)
        return self.regressor(torch.cat([img_feat, s_emb, d_emb], dim=1))

# ===========================
# 3. DATASET (HF 資料庫防呆讀取)
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
        try: t = torch.tensor([item.get('SPEI_30d',0), item.get('SPEI_1y',0), item.get('SPEI_2y',0)], dtype=torch.float32)
        except: t = torch.zeros(3, dtype=torch.float32)
        
        return image, s_idx, d_idx, t

# ===========================
# 4. TRAINING LOOP
# ===========================
def train_fold(fold, train_hf, val_hf, s_map, d_map):
    print(f"\n🚀 Fold {fold} | Size: {CONFIG['img_size']} | Batch: {CONFIG['batch_size']} | Unfreeze: {CONFIG['unfreeze_layers']}")
    
    tf_train = transforms.Compose([
        transforms.Resize((int(CONFIG['img_size']*1.14), int(CONFIG['img_size']*1.14))), 
        transforms.RandomCrop((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tf_val = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    loader_train = DataLoader(HFBeetleDataset(train_hf, s_map, d_map, tf_train), batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    loader_val = DataLoader(HFBeetleDataset(val_hf, s_map, d_map, tf_val), batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)

    # 1. 載入模型並瘦身 (省顯存)
    base_clip = open_clip.create_model('hf-hub:imageomics/bioclip-2', pretrained=True)
    if hasattr(base_clip, 'transformer'): del base_clip.transformer
    if hasattr(base_clip, 'token_embedding'): del base_clip.token_embedding
    
    model = BioClip2_HybridRegressor(base_clip, len(s_map), len(d_map), n_last_trainable_resblocks=CONFIG['unfreeze_layers']).to(device)
    
    # 2. [高分關鍵] 雙學習率設置
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if 'bioclip' in name: backbone_params.append(p)
        else: head_params.append(p)
            
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': CONFIG['lr_backbone']}, # 解凍層用小 LR
        {'params': head_params, 'lr': CONFIG['lr_head']}          # 新接的層用大 LR
    ], weight_decay=1e-3)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    criterion = nn.MSELoss(reduction='none') 

    best_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        loop = tqdm(loader_train, desc=f"Ep {epoch+1}", leave=False)
        for img, s, d, t in loop:
            img, s, d, t = img.to(device), s.to(device), d.to(device), t.to(device)
            optimizer.zero_grad()
            out = model(img, s, d)
            
            # 3. [高分關鍵] 極端值加權 Loss
            raw_loss = criterion(out[:, :3], t)
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
            torch.save(model.state_dict(), f"{CONFIG['save_dir']}/model_bioclip_{CONFIG['img_size']}_fold_{fold}.pth")
            
    print(f"✅ Fold {fold} Best MSE: {best_loss:.4f}")
    del model, optimizer, base_clip; torch.cuda.empty_cache(); gc.collect()

def main():
    try:
        with open(CONFIG['mapping_path'], "r") as f: m = json.load(f)
    except:
        m = {'species_to_idx':{}, 'domain_to_idx':{}}
        
    ds = load_from_disk(CONFIG['dataset_path'])
    if isinstance(ds, dict) or "DatasetDict" in str(type(ds)):
        ds = concatenate_datasets([ds[k] for k in ds.keys()])
        
    kfold = KFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
    for fold, (t_idx, v_idx) in enumerate(kfold.split(np.arange(len(ds)))):
        train_fold(fold, ds.select(t_idx), ds.select(v_idx), m.get('species_to_idx',{}), m.get('domain_to_idx',{}))

if __name__ == '__main__': main()
