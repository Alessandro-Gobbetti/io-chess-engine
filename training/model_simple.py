"""
Simplified ChessNet with WDL+Mate output.

Same architecture as model_repvgg.py but outputs WDL + mate instead of tanh.
Used for Phase 1 validation to ensure the backbone works with WDL targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_repvgg import RepVGGBlock


class ChessNetWDL(nn.Module):
    """
    Same encoder as ChessNet (model_repvgg.py), but with WDL+mate output.
    
    Output:
        wdl: [B, 3] Win/Draw/Loss probabilities (softmax)
        mate: [B, 1] Mate distance (sigmoid, 0=no mate, 1=M1)
    """
    def __init__(self, n_filters=64, n_blocks=8, n_globals=15):
        super().__init__()
        self.n_filters = n_filters
        
        # === ENCODER (same as model_repvgg.py) ===
        # Stem: Board planes (32 channels)
        self.stem = nn.Conv2d(32, n_filters, 3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(n_filters)
        
        # Stem: Global features
        self.stem_global = nn.Linear(n_globals, n_filters, bias=False)
        self.bn_global = nn.BatchNorm1d(n_filters)
        
        self.stem_act = nn.ReLU(inplace=True)
        
        # Body: RepVGG blocks (same as original)
        self.body = nn.Sequential(*[RepVGGBlock(n_filters) for _ in range(n_blocks)])
        
        # === HEAD (modified for WDL+mate) ===
        self.head_conv = nn.Conv2d(n_filters, 32, 1)
        self.head_act = nn.ReLU(inplace=True)
        self.head_flat = nn.Flatten()
        self.head_hidden = nn.Linear(32 * 64, 128)
        self.head_act2 = nn.ReLU(inplace=True)
        
        # WDL + Mate outputs
        self.head_wdl = nn.Linear(128, 3)   # -> softmax
        self.head_mate = nn.Linear(128, 1)  # -> sigmoid
        
    def forward(self, layers, global_v):
        """
        Args:
            layers: [B, 32, 8, 8] float32 normalized board planes (0-1)
            global_v: [B, n_globals] float32 global features
            
        Returns:
            wdl: [B, 3] Win/Draw/Loss probabilities
            mate: [B, 1] Mate distance (0-1)
        """
        # Process board
        x = self.bn_stem(self.stem(layers))
        
        # Process globals
        x_g = self.bn_global(self.stem_global(global_v))
        x_g = x_g.view(-1, self.n_filters, 1, 1)
        
        # Fuse: board + global context
        x = self.stem_act(x + x_g)
        
        # Body
        x = self.body(x)
        
        # Head
        x = self.head_conv(x)
        x = self.head_act(x)
        x = self.head_flat(x)
        x = self.head_hidden(x)
        x = self.head_act2(x)
        
        # Outputs
        wdl = F.softmax(self.head_wdl(x), dim=1)
        mate = torch.sigmoid(self.head_mate(x))
        
        return wdl, mate
    
    def fuse_model(self):
        """Fuse BN for inference (RepVGG style)."""
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
        # Fuse stem BN layers
        torch.quantization.fuse_modules(self, [['stem', 'bn_stem']], inplace=True)
        torch.quantization.fuse_modules(self, [['stem_global', 'bn_global']], inplace=True)


# =============================================================================
#   SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- ChessNetWDL Check on {device} ---")
    
    model = ChessNetWDL(n_filters=64, n_blocks=6).to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    B = 2
    dummy_layers = torch.randn(B, 32, 8, 8).to(device)
    dummy_global = torch.randn(B, 15).to(device)
    
    with torch.no_grad():
        wdl, mate = model(dummy_layers, dummy_global)
    
    print(f"Input shapes: layers={dummy_layers.shape}, global={dummy_global.shape}")
    print(f"Output shapes: wdl={wdl.shape}, mate={mate.shape}")
    print(f"WDL sample: {wdl[0].cpu().numpy()}")
    print(f"WDL sum: {wdl.sum(dim=1)}")  # Should be ~1.0
    print(f"Mate sample: {mate[0].item():.4f}")
    print("Status: OK")
