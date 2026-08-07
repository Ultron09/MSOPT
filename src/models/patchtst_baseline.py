"""
PatchTST PyTorch Baseline Model (Nie et al., ICLR 2023)
=========================================================
Implements subseries patching + ViT Transformer backbone for time series direction classification.

Protocol:
- Non-overlapping patches of length P=16, stride S=8
- Linear patch embedding into D_model=64
- 2-Layer Transformer Encoder with Multi-Head Self-Attention
- Head classifier predicting Fork B directional target y_dir in {-1, 0, 1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchTSTBaselineClassifier(nn.Module):
    """
    Subseries Patch Transformer for Time Series Classification.
    """
    def __init__(
        self,
        seq_len: int = 60,
        patch_len: int = 16,
        stride: int = 8,
        num_classes: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        
        # Calculate number of patches: N = (seq_len - patch_len) // stride + 1
        self.num_patches = max(1, (seq_len - patch_len) // stride + 1)
        
        # 1. Linear Patch Projection
        self.patch_embed = nn.Linear(patch_len, d_model)
        
        # 2. Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classifier Head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x [Batch, Seq_Len]
        Output: Logits [Batch, Num_Classes]
        """
        B, L = x.shape
        
        # Extract patches via unfolding: [B, Num_Patches, Patch_Len]
        if L < self.patch_len:
            x = F.pad(x, (0, self.patch_len - L))
            L = self.patch_len
            
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride) # [B, N, P]
        
        # Project patches: [B, N, D]
        enc_in = self.patch_embed(patches) + self.pos_embed[:, :patches.shape[1], :]
        
        # Transformer Attention: [B, N, D]
        enc_out = self.transformer(enc_in)
        
        # Mean Pooling across patches
        pooled = enc_out.mean(dim=1)
        
        # Class Logits
        return self.head(pooled)


if __name__ == "__main__":
    B, L = 16, 60
    dummy_x = torch.randn(B, L)
    model = PatchTSTBaselineClassifier(seq_len=L, patch_len=16, stride=8, num_classes=3)
    logits = model(dummy_x)
    
    print("[PatchTST PyTorch Baseline Check]")
    print(f"  Input Series Shape: {dummy_x.shape}")
    print(f"  Number of Patches: {model.num_patches}")
    print(f"  Output Logits Shape: {logits.shape} (Expected: [{B}, 3])")
