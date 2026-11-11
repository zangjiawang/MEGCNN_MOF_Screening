"""
Note to Reviewers: 
This repository contains a minimal working example with 3 MOF samples. 
Because the dataset is restricted here, the K-Fold and LOMO validation will throw errors. 
This script is intended to demonstrate the graph construction, dual-input architecture, and memory mechanism.
"""
import os
import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局设置
# ============================================================
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

EXCEL_PATH = "data/sample_data.xlsx"
CIF_DIR = r"relaxed_structures" 

# 图构建参数
CUTOFF = 4.0          
MAX_NEIGHBORS = 8    
EDGE_DIM = 32         
HIDDEN_DIM = 64
MEM_SLOTS = 16        
MEM_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 4
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 30

# ============================================================
# 1. 读取 Excel，匹配 CIF 并提取描述符
# ============================================================
def load_metadata(excel_path, cif_dir):
    xls = pd.ExcelFile(excel_path)
    df = None
    for sheet in xls.sheet_names:
        temp_df = pd.read_excel(xls, sheet_name=sheet)
        for col in temp_df.columns:
            if 'mof_id' in str(col).strip().lower():
                df = temp_df
                print(f"成功在 sheet '{sheet}' 中找到列名: '{col}'")
                break
        if df is not None:
            break
            
    if df is None:
        df = pd.read_excel(xls, sheet_name=0)

    mof_id_col = None
    for col in df.columns:
        if 'mof' in str(col).lower() and 'id' in str(col).lower():
            mof_id_col = col
            break
    if mof_id_col is None:
        mof_id_col = df.columns[0]
        
    print(f"最终使用的 ID 列名为: '{mof_id_col}'")
    
    # ==== 新增：提取第 3 到 13 列作为描述符 ====
    # pandas 索引从 0 开始，第 3 列到第 13 列的切片是 2:13
    desc_cols = df.columns[2:13] 
    print("Descriptor columns:", list(desc_cols))
    
    # 目标列取最后三列 (GOH, GO, GOOH)
    target_cols = df.columns[-3:]
    print("Target columns:", list(target_cols))

    records = []
    missing = []
    
    for _, row in df.iterrows():
        mof_id_val = row[mof_id_col]
        if pd.isna(mof_id_val): 
            continue
        mof_id = str(mof_id_val).strip()
        
        cif_path = os.path.join(cif_dir, f"{mof_id}.cif")
        if not os.path.exists(cif_path):
            alt = os.path.join(cif_dir, f"{mof_id}.CIF")
            if os.path.exists(alt):
                cif_path = alt
            else:
                missing.append(mof_id)
                continue
                
        # ==== 修改：提取描述符并处理潜在的 NaN ====
        desc_vals = row[desc_cols].values.astype(np.float32)
        desc_vals = np.nan_to_num(desc_vals, nan=0.0) # 将缺失值替换为 0
        
        y = row[target_cols].values.astype(np.float32)
        records.append({
            'mof_id': mof_id,
            'cif_path': cif_path,
            'descriptors': desc_vals, # ==== 新增 ====
            'targets': y
        })
        
    print(f"匹配结果: 共 {len(df)} 行，成功匹配到 {len(records)} 个 CIF 文件。")
    return records

# ============================================================
# 2. 从 CIF 构建晶体图
# ============================================================
def gaussian_expand(distances, centers, width=0.5):
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))

def build_graph_from_cif(cif_path, element_to_idx, cutoff=CUTOFF, max_neighbors=MAX_NEIGHBORS):
    try:
        structure = Structure.from_file(cif_path)
    except Exception as e:
        return None

    elements = [site.specie.symbol for site in structure]
    node_indices = [element_to_idx[e] for e in elements]
    x = torch.tensor(node_indices, dtype=torch.long)  

    neighbors = structure.get_all_neighbors(r=cutoff)
    rows, cols, dists = [], [], []
    for i, nbrs in enumerate(neighbors):
        nbrs_sorted = sorted(nbrs, key=lambda n: n.nn_distance)[:max_neighbors]
        for nbr in nbrs_sorted:
            j = nbr.index
            d = nbr.nn_distance
            rows.append(i); cols.append(j); dists.append(d)
            rows.append(j); cols.append(i); dists.append(d)

    if len(rows) == 0:
        n = len(elements)
        rows = list(range(n)); cols = list(range(n)); dists = [0.0] * n

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_dist = torch.tensor(dists, dtype=torch.float32)
    centers = torch.linspace(0, 6, EDGE_DIM)
    edge_attr = gaussian_expand(edge_dist, centers)

    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}

# ============================================================
# 3. Dataset 与 collate
# ============================================================
class CrystalGraphDataset(Dataset):
  
    def __init__(self, graphs, descriptors, targets):
        self.graphs = graphs
        self.descriptors = descriptors
        self.targets = targets

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        d = torch.tensor(self.descriptors[idx], dtype=torch.float32) # ==== 新增 ====
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return g['x'], g['edge_index'], g['edge_attr'], d, y

def collate_fn(batch):

    x_list, ei_list, ea_list, d_list, y_list = zip(*batch)
    x = torch.cat(x_list, dim=0)
    d = torch.stack(d_list, dim=0) 
    y = torch.stack(y_list, dim=0)

    batch_vec = []
    edge_indices = []
    edge_attrs = []
    offset = 0
    for i, (x_i, ei, ea) in enumerate(zip(x_list, ei_list, ea_list)):
        batch_vec.append(torch.full((x_i.size(0),), i, dtype=torch.long))
        edge_indices.append(ei + offset)
        edge_attrs.append(ea)
        offset += x_i.size(0)

    edge_index = torch.cat(edge_indices, dim=1)
    edge_attr = torch.cat(edge_attrs, dim=0)
    batch_vec = torch.cat(batch_vec, dim=0)

    return x, edge_index, edge_attr, d, y, batch_vec

# ============================================================
# 4. 图卷积层 & 外部记忆模块
# ============================================================
class GNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dropout=0.2):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        msg = self.msg_mlp(torch.cat([x[src], x[dst], edge_attr], dim=-1))
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        out = self.update_mlp(torch.cat([x, agg], dim=-1))
        return self.norm(x + out) 

class ExternalMemory(nn.Module):
    def __init__(self, mem_slots, mem_dim, query_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_slots, mem_dim) * 0.1)
        self.q_proj = nn.Linear(query_dim, mem_dim)
        self.k_proj = nn.Linear(mem_dim, mem_dim)
        self.v_proj = nn.Linear(mem_dim, mem_dim)
        self.scale = math.sqrt(mem_dim)

    def forward(self, g):
        q = self.q_proj(g)                    
        k = self.k_proj(self.memory)          
        v = self.v_proj(self.memory)          
        attn = torch.softmax(q @ k.T / self.scale, dim=-1)  
        read = attn @ v                       
        return read, attn

# ============================================================
# 5. MEGCNN 主模型
# ============================================================
class MEGCNN(nn.Module):
    def __init__(self, num_elements, hidden_dim=64, edge_dim=32,
                 mem_slots=16, mem_dim=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_elements, hidden_dim)
        self.layers = nn.ModuleList([
            GNNLayer(hidden_dim, edge_dim, dropout) for _ in range(num_layers)
        ])
        self.memory = ExternalMemory(mem_slots, mem_dim, hidden_dim)
        
    
        self.desc_proj = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
       
        self.fusion = nn.Linear(hidden_dim * 2 + mem_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            ) for _ in range(3)
        ])


    def forward(self, x, edge_index, edge_attr, batch, desc):
        h = self.embedding(x)  
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        # 全局平均池化
        num_graphs = batch.max().item() + 1
        g = torch.zeros(num_graphs, h.size(1), device=h.device)
        g.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).float().unsqueeze(1)
        g = g / counts  

        # 外部记忆读操作
        read, attn = self.memory(g)  

        d_feat = self.desc_proj(desc)

      
        z = self.fusion(torch.cat([g, read, d_feat], dim=-1))
        z = self.dropout(z)

        # 多任务输出
        outputs = [head(z) for head in self.heads]
        return torch.cat(outputs, dim=1), attn

# ============================================================
# 6. 训练与评估
# ============================================================
# ============================================================
# 7. 核心：重构训练与评估流程，支持 5-Fold CV 和 LOMO
# ============================================================

def train_and_evaluate(train_idx, val_idx, graphs, targets, descriptors, elements):
    """通用的训练和评估函数，用于单次划分、CV 和 LOMO"""
    # 1. 提取数据
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    train_targets = targets[train_idx]
    val_targets = targets[val_idx]
    
    # 2. 描述符标准化（关键：只在训练集上 fit）
    scaler = StandardScaler()
    train_desc = scaler.fit_transform(descriptors[train_idx])
    val_desc = scaler.transform(descriptors[val_idx])

    # 3. DataLoader
    train_dataset = CrystalGraphDataset(train_graphs, train_desc, train_targets)
    val_dataset = CrystalGraphDataset(val_graphs, val_desc, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4. 初始化模型（每次重新初始化）
    model = MEGCNN(
        num_elements=len(elements), hidden_dim=HIDDEN_DIM, edge_dim=EDGE_DIM,
        mem_slots=MEM_SLOTS, mem_dim=MEM_DIM, num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    # 5. 训练循环
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, mae, rmse, r2, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    # 6. 加载最佳模型并在验证/测试集上评估
    model.load_state_dict(best_state)
    _, mae, rmse, r2, preds, targets_val = evaluate(model, val_loader, criterion)
    return mae, rmse, r2, preds, targets_val

def main():
    records = load_metadata(EXCEL_PATH, CIF_DIR)
    if len(records) == 0: return

    # 提取元素字典
    element_set = set()
    for rec in tqdm(records, desc="Scanning elements"):
        try:
            s = Structure.from_file(rec['cif_path'])
            element_set.update([site.specie.symbol for site in s])
        except Exception: pass
    elements = sorted(element_set)

    # 构建图、描述符和目标
    print("Building crystal graphs...")
    graphs, targets, descriptors, used_ids = [], [], [], []
    for rec in tqdm(records):
        g = build_graph_from_cif(rec['cif_path'], {e: i for i, e in enumerate(elements)})
        if g is None: continue
        graphs.append(g)
        targets.append(rec['targets'])
        descriptors.append(rec['descriptors'])
        used_ids.append(rec['mof_id'])
        
    targets = np.array(targets, dtype=np.float32)
    descriptors = np.array(descriptors, dtype=np.float32)
    print(f"Successfully built {len(graphs)} graphs.\n")

    # ========================================================
    # 任务 1：5 折交叉验证 (5-Fold Cross Validation)
    # ========================================================
    print("=" * 60)
    print("Running 5-Fold Cross Validation")
    print("=" * 60)
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_r2_list = []
    cv_mae_list = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(graphs)):
        print(f"--- Fold {fold+1} / 5 ---")
        mae, rmse, r2, _, _ = train_and_evaluate(train_idx, val_idx, graphs, targets, descriptors, elements)
        print(f"Fold {fold+1} Results: GOH R2={r2[0]:.4f}, GO R2={r2[1]:.4f}, GOOH R2={r2[2]:.4f}")
        cv_r2_list.append(r2)
        cv_mae_list.append(mae)
        
    cv_r2_array = np.array(cv_r2_list)
    cv_mae_array = np.array(cv_mae_list)
    print("\n5-Fold CV Summary (Mean ± Std):")
    for i, name in enumerate(['GOH', 'GO', 'GOOH']):
        print(f"{name}: R2 = {np.mean(cv_r2_array[:, i]):.4f} ± {np.std(cv_r2_array[:, i]):.4f}, "
              f"MAE = {np.mean(cv_mae_array[:, i]):.4f} ± {np.std(cv_mae_array[:, i]):.4f}")

    # ========================================================
    # 任务 2：留一金属中心验证 (LOMO, Leave-One-Metal-Out)
    # ========================================================
    print("\n" + "=" * 60)
    print("Running Leave-One-Metal-Out (LOMO) Validation")
    print("=" * 60)
    
    # 选取数据集中出现频率最高的 4 个过渡金属
    from collections import Counter
    metals = []
    for rec in records:
        s = Structure.from_file(rec['cif_path'])
        found_metal = None
        for site in s:
            if site.specie.symbol in ['Fe', 'Co', 'Ni', 'Cu']:
                found_metal = site.specie.symbol
                break
        metals.append(found_metal)
        
    # 将 'None' 替换为其他，以免报错
    metals = np.array([m if m else "Other" for m in metals])
    
    for target_metal in ['Fe', 'Co', 'Ni', 'Cu']:
        print(f"\n--- LOMO: Leave out {target_metal} ---")
        # 找出包含目标金属的索引作为测试集
        test_idx = np.where(metals == target_metal)[0]
        train_idx = np.where(metals != target_metal)[0]
        
        if len(test_idx) < 5:
            print(f"Not enough samples for {target_metal} (n={len(test_idx)}). Skipping.")
            continue
            
        print(f"Train samples: {len(train_idx)}, Test samples ({target_metal}): {len(test_idx)}")
        mae, rmse, r2, _, _ = train_and_evaluate(train_idx, test_idx, graphs, targets, descriptors, elements)
        print(f"LOMO {target_metal} Results:")
        print(f"  GOH: R2={r2[0]:.4f}, MAE={mae[0]:.4f} eV")
        print(f"  GO:  R2={r2[1]:.4f}, MAE={mae[1]:.4f} eV")
        print(f"  GOOH: R2={r2[2]:.4f}, MAE={mae[2]:.4f} eV")

if __name__ == "__main__":
    main()
