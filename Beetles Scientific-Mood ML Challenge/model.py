import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import sys
import pandas as pd
import numpy as np
import math
from PIL import Image
import gc

# ==========================================
# V28.0: Grandmaster JIT Ensemble (Universal Lazy Loader)
# ==========================================
print("==========================================")
print("!!! DEBUG: MODEL SCRIPT V28.0 (JIT Lazy Loader) LOADED !!!")
print("==========================================")

sys.path.append(os.path.dirname(__file__))

# 嘗試載入必要的庫
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# ==========================================
# 1. 模型架構定義 (BioClip & ConvNeXt)
# ==========================================

# --- A. BioClip 架構 ---
class BioClip2_HybridRegressor(nn.Module):
    def __init__(self, bioclip, num_species=152, num_domains=10, 
                 vision_dim=768, species_emb_dim=64, domain_emb_dim=16, 
                 hidden_size=512, num_outputs=6):
        super().__init__()
        self.bioclip = bioclip
        self.species_embedding = nn.Embedding(num_species, species_emb_dim, padding_idx=0)
        self.domain_embedding = nn.Embedding(num_domains, domain_emb_dim, padding_idx=0)
        
        total_input_dim = vision_dim + species_emb_dim + domain_emb_dim
        
        self.regressor = nn.Sequential(
            nn.Linear(total_input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_outputs),
        )

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
        if self.bioclip is None: return torch.zeros((x.shape[0], 768), device=x.device)
        try:
            x = self.bioclip.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            _cls = self.bioclip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([_cls, x], dim=1)
            pos_embed = self.bioclip.visual.positional_embedding.to(x.dtype)
            if pos_embed.shape[0] != x.shape[1]: pos_embed = self._resize_pos_embed(pos_embed, x.shape[1])
            x = x + pos_embed
            if hasattr(self.bioclip.visual, 'ln_pre'): x = self.bioclip.visual.ln_pre(x)
            if not self.bioclip.visual.transformer.batch_first: x = x.transpose(0, 1).contiguous()
            for r in self.bioclip.visual.transformer.resblocks: x = r(x, attn_mask=None)
            if not self.bioclip.visual.transformer.batch_first: x = x.transpose(0, 1)
            pooled, _ = self.bioclip.visual._pool(x)
            if self.bioclip.visual.proj is not None: pooled = pooled @ self.bioclip.visual.proj
            return F.normalize(pooled, dim=-1)
        except: return torch.zeros((x.shape[0], 768), device=x.device)

    def forward(self, images, species_idx, domain_idx):
        return self.regressor(torch.cat([self.forward_image(images), self.species_embedding(species_idx), self.domain_embedding(domain_idx)], dim=1))

# --- B. ConvNeXt 架構 ---
class ConvNext_HybridRegressor(nn.Module):
    def __init__(self, vision_dim=1024, num_species=152, num_domains=10, 
                 species_emb_dim=64, domain_emb_dim=16, hidden_size=512, num_outputs=6):
        super().__init__()
        if TIMM_AVAILABLE:
            self.backbone = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=False, num_classes=0)
        else:
            print("WARNING: Timm not found, using dummy backbone")
            self.backbone = None # Should fallback to torchvision if needed, but keeping simple
            
        self.species_embedding = nn.Embedding(num_species, species_emb_dim, padding_idx=0)
        self.domain_embedding = nn.Embedding(num_domains, domain_emb_dim, padding_idx=0)
        
        total_input_dim = vision_dim + species_emb_dim + domain_emb_dim
        self.regressor = nn.Sequential(
            nn.Linear(total_input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(), nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_outputs),
        )

    def forward_image(self, x):
        if self.backbone is None: return torch.zeros((x.shape[0], 1024), device=x.device)
        return F.normalize(self.backbone(x), dim=-1)

    def forward(self, images, species_idx, domain_idx):
        return self.regressor(torch.cat([self.forward_image(images), self.species_embedding(species_idx), self.domain_embedding(domain_idx)], dim=1))

# ==========================================
# 2. Model 類別 (JIT Lazy Loader)
# ==========================================
class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEBUG: Initializing JIT Ensemble on {self.device}...")
        self.model_files = [] # 只存檔名，不存模型實體
        self._load_mappings()
        self._init_transforms()

    def _load_mappings(self):
        try:
            with open(os.path.join(os.path.dirname(__file__), 'mappings.json'), 'r') as f:
                mappings = json.load(f)
            self.species_to_idx = mappings.get("species_to_idx", {})
            self.domain_to_idx = mappings.get("domain_to_idx", {})
        except:
            self.species_to_idx = {}
            self.domain_to_idx = {}

    def _init_transforms(self):
        from torchvision import transforms
        # 通用預處理：使用 448 (對 BioClip 好，對 ConvNeXt 384 也相容)
        # 如果您想追求 ConvNeXt 極致，可以改成 384，但 BioClip 可能會稍微掉分
        # 這裡折衷使用 448，因為 ConvNeXt 吃大圖通常沒問題
        self.preprocess = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load(self):
        print("DEBUG: Scanning for models...")
        base_dir = os.path.dirname(__file__)
        # 只搜尋 .pth 檔案，完全不載入權重
        self.model_files = sorted([f for f in os.listdir(base_dir) if f.endswith(".pth") and "model" in f])
        
        if not self.model_files:
            print("CRITICAL: No model weights found!")
            # Fallback path if explicit path exists
            if os.path.exists(os.path.join(base_dir, "model.pth")):
                self.model_files = ["model.pth"]
        
        print(f"DEBUG: Found {len(self.model_files)} model files. (Lazy Loading enabled)")

    def predict(self, inputs):
        if isinstance(inputs, pd.DataFrame): datapoints = inputs.to_dict(orient='records')
        else: datapoints = inputs

        ensemble_outputs = [] 
        default_img = Image.new('RGB', (448, 448), (0, 0, 0))

        # ==========================================
        # JIT Loop: Load -> Predict -> Delete
        # ==========================================
        base_dir = os.path.dirname(__file__)
        
        for fname in self.model_files:
            fold_outputs = []
            weight_path = os.path.join(base_dir, fname)
            
            # --- 1. 動態建立模型 ---
            # 根據檔名決定架構
            try:
                if "convnext" in fname.lower():
                    # ConvNeXt 模式
                    model = ConvNext_HybridRegressor(vision_dim=1024) # Base=1024
                    print(f"   [JIT] Running ConvNeXt: {fname}")
                else:
                    # BioClip 模式
                    import open_clip
                    base_bioclip = open_clip.create_model('hf-hub:imageomics/bioclip-2', pretrained=None)
                    # 瘦身: 刪除 Text Encoder
                    if hasattr(base_bioclip, 'transformer'): del base_bioclip.transformer
                    if hasattr(base_bioclip, 'token_embedding'): del base_bioclip.token_embedding
                    model = BioClip2_HybridRegressor(base_bioclip, n_last_trainable_resblocks=2)
                    print(f"   [JIT] Running BioClip: {fname}")
                
                # --- 2. 載入權重 ---
                state_dict = torch.load(weight_path, map_location=self.device)
                clean_state_dict = {k.replace('module.', '').replace('net.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(clean_state_dict, strict=False)
                
                model.to(self.device)
                model.eval()
                
                # --- 3. 預測 ---
                with torch.no_grad():
                    for entry in datapoints:
                        try:
                            img_input = entry.get('relative_img', entry.get('file_path'))
                            image = default_img
                            if isinstance(img_input, str) and os.path.exists(img_input):
                                try: image = Image.open(img_input).convert("RGB")
                                except: pass
                            elif isinstance(img_input, Image.Image): image = img_input.convert("RGB")
                            
                            # TTA: Original + Flip
                            s_name = entry.get('scientificName', '<UNK>')
                            d_id = str(entry.get('domainID', '0'))
                            s_idx = self.species_to_idx.get(s_name, 0)
                            d_idx = self.domain_to_idx.get(d_id, 0)
                            s_tensor = torch.tensor([s_idx], device=self.device)
                            d_tensor = torch.tensor([d_idx], device=self.device)

                            # 1. Original
                            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                            out1 = model(img_tensor, s_tensor, d_tensor)
                            
                            # 2. Flip
                            img_flip = self.preprocess(image.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(self.device)
                            out2 = model(img_flip, s_tensor, d_tensor)
                            
                            output = (out1 + out2) / 2.0
                            fold_outputs.append(output.cpu())
                        except:
                            fold_outputs.append(torch.zeros(1, 6))
                
                # --- 4. 關鍵：刪除模型釋放記憶體 ---
                del model
                if "base_bioclip" in locals(): del base_bioclip
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"CRITICAL ERROR processing {fname}: {e}")
                # 出錯補零，避免中斷
                fold_outputs = [torch.zeros(1, 6) for _ in datapoints]

            if len(fold_outputs) > 0:
                ensemble_outputs.append(torch.cat(fold_outputs, dim=0))
            else:
                ensemble_outputs.append(torch.zeros(len(datapoints), 6))

        if not ensemble_outputs:
             return {"SPEI_30d": {"mu":0.0, "sigma":1.0}, "SPEI_1y": {"mu":0.0, "sigma":1.0}, "SPEI_2y": {"mu":0.0, "sigma":1.0}}

        # --- 聚合邏輯 (Aggregation) ---
        stacked_preds = torch.stack(ensemble_outputs, dim=0) 
        mus = stacked_preds[:, :, :3]
        log_vars = stacked_preds[:, :, 3:]
        vars_aleatoric = torch.exp(log_vars)

        final_mu_per_img = torch.mean(mus, dim=0)
        term1 = torch.mean(vars_aleatoric, dim=0)
        term2 = torch.var(mus, dim=0, unbiased=False)
        final_std_per_img = torch.sqrt(term1 + term2)

        agg_mu = torch.mean(final_mu_per_img, dim=0)
        mean_prediction_uncertainty = torch.mean(final_std_per_img, dim=0)
        
        if final_mu_per_img.size(0) > 1:
            data_spread_uncertainty = torch.std(final_mu_per_img, dim=0)
        else:
            data_spread_uncertainty = torch.zeros_like(agg_mu)
            
        agg_sigma = mean_prediction_uncertainty + data_spread_uncertainty

        mu_list = agg_mu.tolist()
        sigma_list = agg_sigma.tolist()
        mu_list = [m if math.isfinite(m) else 0.0 for m in mu_list]
        sigma_list = [s if math.isfinite(s) else 1.0 for s in sigma_list]

        return {
            "SPEI_30d": {"mu": mu_list[0], "sigma": sigma_list[0]},
            "SPEI_1y":  {"mu": mu_list[1], "sigma": sigma_list[1]},
            "SPEI_2y":  {"mu": mu_list[2], "sigma": sigma_list[2]},
        }