import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import r2_score
from open_clip import create_model_and_transforms
from argparse import ArgumentParser
from pathlib import Path

# --- 1. 全局字典載入 ---
try:
    base_path = Path(__file__).resolve().parent
    mapping_path = base_path / "mappings.json"
    if not mapping_path.exists():
        mapping_path = Path("mappings.json")

    with open(mapping_path, "r") as f:
        MAPPINGS = json.load(f)
    SPECIES_TO_IDX = MAPPINGS["species_to_idx"]
    DOMAIN_TO_IDX = MAPPINGS["domain_to_idx"]
    print(f"✅ [Utils] 成功載入詞彙表: {len(SPECIES_TO_IDX)} 種物種, {len(DOMAIN_TO_IDX)} 個氣候域")
except Exception as e:
    print(f"⚠️ [Utils] 警告: 無法載入 mappings.json ({e})")
    SPECIES_TO_IDX = {"<UNK>": 0}
    DOMAIN_TO_IDX = {"0": 0}

# --- 2. 核心 Dataset 類別 ---
class BeetleDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, is_train=True):
        self.dataset = hf_dataset
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["file_path"].convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        s_name = item.get("scientificName", "<UNK>")
        d_id = str(item.get("domainID", "0"))
        s_idx = SPECIES_TO_IDX.get(s_name, SPECIES_TO_IDX.get("<UNK>", 0))
        d_idx = DOMAIN_TO_IDX.get(d_id, DOMAIN_TO_IDX.get("0", 0))
        
        if self.is_train:
            labels = torch.tensor([
                item.get("SPEI_30d", 0.0),
                item.get("SPEI_1y", 0.0),
                item.get("SPEI_2y", 0.0)
            ], dtype=torch.float32)
        else:
            labels = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        
        return image, torch.tensor(s_idx, dtype=torch.long), torch.tensor(d_idx, dtype=torch.long), labels

# --- 3. Data Augmentation ---
def get_bioclip_transforms(is_train=True, img_size=336): # <--- 預設改為 336
    # BioClip/CLIP 標準化參數
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    
    if is_train:
        # 訓練時：強大的資料增強
        return transforms.Compose([
            # [修改] 解析度提升
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)), 
            transforms.RandomHorizontalFlip(), 
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # 驗證/測試時：只做標準化
        return transforms.Compose([
            # [修改] 解析度提升
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

# --- 4. Early Stopping 機制 [新增] ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): 容忍多少個 epoch loss 沒有下降
            min_delta (float): loss 變化的最小閾值
            path (str): 模型儲存路徑
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            return True # 這次有進步
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def save_checkpoint(self, model):
        # 呼叫 model 內部的 save_parameters 方法
        model.save_parameters(self.path)

# --- 5. 其他工具 ---
def get_bioclip():
    print("正在載入 BioClip 模型...")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"   [Utils] Detected device: {device}")
    model, _, _ = create_model_and_transforms("hf-hub:imageomics/bioclip-2", output_dict=True, require_pretrained=True)
    return model.to(device)

def evalute_spei_r2_scores(gts, preds):
    if isinstance(gts, torch.Tensor): gts = gts.cpu().numpy()
    if isinstance(preds, torch.Tensor): preds = preds.cpu().numpy()
    spei_30_r2 = r2_score(gts[:, 0], preds[:, 0])
    spei_1y_r2 = r2_score(gts[:, 1], preds[:, 1])
    spei_2y_r2 = r2_score(gts[:, 2], preds[:, 2])
    return spei_30_r2, spei_1y_r2, spei_2y_r2

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    species = torch.stack([item[1] for item in batch])
    domains = torch.stack([item[2] for item in batch])
    labels = torch.stack([item[3] for item in batch])
    return images, species, domains, labels

def get_training_args():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader Workers")
    parser.add_argument("--epochs", type=int, default=30, help="最大訓練輪數")
    parser.add_argument("--n_last_trainable_blocks", type=int, default=1, help="解凍層數 (1層較穩)")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace Token")
    parser.add_argument("--k_folds", type=int, default=5, help="K-Fold 數量")
    return parser.parse_args()
