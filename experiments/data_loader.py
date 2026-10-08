"""
Data loading and preprocessing for NiCrC thin film dataset.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os

class ThinFilmDataset(Dataset):
    """NiCrC thin film property dataset."""
    
    PROP_NAMES = ['Sa_nm', 'H_GPa', 'E_GPa', 'stress_GPa', 'H_Estar']
    FEAT_NAMES = ['C2H2_sccm', 'bias_V', 'time_s', 'material']
    
    def __init__(self, data_path, feature_noise=0.0, target_noise=0.0):
        """
        Args:
            data_path: path to training_dataset.csv
            feature_noise: Gaussian noise std for data augmentation
            target_noise: Gaussian noise std for data augmentation
        """
        self.df = pd.read_csv(data_path)
        self.feature_noise = feature_noise
        self.target_noise = target_noise
        
        # Encode material type (NiCrC=0, HEA=1)
        self.df['material'] = (self.df['material'] == 'HEA').astype(float)
        
        # Handle NaN features: fill with column mean before normalization
        self.features = self.df[self.FEAT_NAMES].values.astype(np.float32)
        for i in range(self.features.shape[1]):
            col = self.features[:, i]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col_mean = np.nanmean(col)
                col[nan_mask] = col_mean
        
        # Targets with NaN handling
        self.targets = self.df[self.PROP_NAMES].values.astype(np.float32)
        
        # Source weights for weighted training
        source_map = {'exact': 3.0, 'synthetic': 0.5, 'digitized': 0.8}
        self.weights = np.array([source_map.get(s, 1.0) for s in self.df['source']])
        
        # Normalize features
        self.feat_mean = np.nanmean(self.features, axis=0)
        self.feat_std = np.nanstd(self.features, axis=0)
        self.feat_std[self.feat_std < 1e-8] = 1.0
        self.features = (self.features - self.feat_mean) / self.feat_std
        
        # Normalize targets (per property using non-NaN values)
        self.tgt_mean = np.zeros(len(self.PROP_NAMES))
        self.tgt_std = np.ones(len(self.PROP_NAMES))
        for i in range(len(self.PROP_NAMES)):
            valid = ~np.isnan(self.targets[:, i])
            if valid.sum() > 0:
                self.tgt_mean[i] = np.nanmean(self.targets[:, i])
                self.tgt_std[i] = np.nanstd(self.targets[:, i])
                if self.tgt_std[i] < 1e-8:
                    self.tgt_std[i] = 1.0
                self.targets[valid, i] = (self.targets[valid, i] - self.tgt_mean[i]) / self.tgt_std[i]
        
        # Track valid targets
        self.valid_mask = ~np.isnan(self.targets)
        self.targets = np.nan_to_num(self.targets, nan=0.0)
        
        print(f'  Loaded {len(self.df)} samples')
        print(f'  Features: {self.FEAT_NAMES}')
        print(f'  Targets: {self.PROP_NAMES}')
        for i, name in enumerate(self.PROP_NAMES):
            n_valid = self.valid_mask[:, i].sum()
            print(f'    {name}: {n_valid} valid, range [{self.tgt_mean[i]-2*self.tgt_std[i]:.2f}, {self.tgt_mean[i]+2*self.tgt_std[i]:.2f}]')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = self.features[idx].copy()
        y = self.targets[idx].copy()
        mask = self.valid_mask[idx].copy()
        w = self.weights[idx]
        
        # Data augmentation
        if self.feature_noise > 0:
            x += np.random.randn(*x.shape) * self.feature_noise
        if self.target_noise > 0:
            valid_idx = np.where(mask)[0]
            y[valid_idx] += np.random.randn(len(valid_idx)) * self.target_noise
        
        return torch.FloatTensor(x), torch.FloatTensor(y), torch.BoolTensor(mask), torch.FloatTensor([w])
    
    def get_normalization(self):
        return {
            'feat_mean': self.feat_mean,
            'feat_std': self.feat_std,
            'tgt_mean': self.tgt_mean,
            'tgt_std': self.tgt_std,
        }


def create_dataloaders(data_path, batch_size=64, train_ratio=0.8, 
                       val_ratio=0.1, seed=42):
    """Create train/val/test dataloaders."""
    dataset = ThinFilmDataset(data_path)
    n = len(dataset)
    
    torch.manual_seed(seed)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    print(f'\nDataset split: train={n_train}, val={n_val}, test={n_test}')
    
    return train_loader, val_loader, test_loader, dataset


def prepare_xgboost_data(data_path):
    """Prepare data for XGBoost baseline."""
    ds = ThinFilmDataset(data_path)
    X = ds.features
    Y = ds.targets
    mask = ds.valid_mask
    tgt_mean = ds.tgt_mean
    tgt_std = ds.tgt_std
    return X, Y, mask, tgt_mean, tgt_std, ds.PROP_NAMES, ds.FEAT_NAMES


if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'unified_dataset', 'training_dataset.csv')
    train_loader, val_loader, test_loader, ds = create_dataloaders(data_path)
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')
    print(f'Test batches: {len(test_loader)}')
    
    x, y, mask, w = next(iter(train_loader))
    print(f'Batch shapes: x={x.shape}, y={y.shape}, mask={mask.shape}')
