"""
Lightweight MoE Model for Chess Evaluation.

Architecture matches model_simple.py so weights can be loaded directly.
4 expert heads: Tactical, Strategic, Major End, Minor End.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_repvgg import RepVGGBlock


class LightExpert(nn.Module):
    """
    Expert head with identical architecture to model_simple.py head.
    This allows loading weights directly from a trained model_simple.
    
    Input: [B, n_filters, 8, 8] spatial features from backbone
    Output: (wdl [B, 3], mate [B, 1])
    """
    def __init__(self, n_filters=64):
        super().__init__()
        self.head_conv = nn.Conv2d(n_filters, 32, 1)
        self.head_act = nn.ReLU(inplace=True)
        self.head_flat = nn.Flatten()
        self.head_hidden = nn.Linear(32 * 64, 128)
        self.head_act2 = nn.ReLU(inplace=True)
        self.head_wdl = nn.Linear(128, 3)
        self.head_mate = nn.Linear(128, 1)
    
    def forward(self, x):
        """
        Args:
            x: [B, n_filters, 8, 8] spatial features
        Returns:
            wdl: [B, 3] win/draw/loss probabilities
            mate: [B, 1] mate distance (0-1)
        """
        x = self.head_act(self.head_conv(x))
        x = self.head_flat(x)
        x = self.head_act2(self.head_hidden(x))
        wdl = F.softmax(self.head_wdl(x), dim=1)
        mate = torch.sigmoid(self.head_mate(x))
        return wdl, mate


class ChessNetLightMoE(nn.Module):
    """
    Lightweight Mixture of Experts model.
    
    Architecture:
        - Backbone: Shared RepVGG encoder (same as model_simple)
        - Experts: 4 LightExpert heads (Tactical, Strategic, Major End, Minor End)
    
    Key feature: Can load weights from trained model_simple to skip Phase 1.
    """
    
    EXPERT_NAMES = ['Tactical', 'Strategic', 'MajorEnd', 'MinorEnd']
    
    def __init__(self, n_filters=64, n_blocks=8, n_globals=15):
        super().__init__()
        self.n_filters = n_filters
        
        # === BACKBONE (identical to model_simple) ===
        self.stem = nn.Conv2d(32, n_filters, 3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(n_filters)
        self.stem_global = nn.Linear(n_globals, n_filters, bias=False)
        self.bn_global = nn.BatchNorm1d(n_filters)
        self.stem_act = nn.ReLU(inplace=True)
        self.body = nn.Sequential(*[RepVGGBlock(n_filters) for _ in range(n_blocks)])
        
        # === EXPERTS (4 heads with identical architecture) ===
        self.experts = nn.ModuleList([
            LightExpert(n_filters) for _ in range(4)
        ])
    
    def forward_backbone(self, layers, global_v):
        """
        Forward through backbone only.
        
        Args:
            layers: [B, 32, 8, 8] board planes
            global_v: [B, n_globals] global features
            
        Returns:
            features: [B, n_filters, 8, 8] spatial features
        """
        x = self.bn_stem(self.stem(layers))
        x_g = self.bn_global(self.stem_global(global_v))
        x_g = x_g.view(-1, self.n_filters, 1, 1)
        x = self.stem_act(x + x_g)
        x = self.body(x)
        return x
    
    def forward(self, layers, global_v, expert_weights=None):
        """
        Forward pass with expert combination.
        
        Args:
            layers: [B, 32, 8, 8] board planes
            global_v: [B, n_globals] global features
            expert_weights: [B, 4] expert weights (softmax). If None, equal weights.
            
        Returns:
            wdl: [B, 3] combined WDL
            mate: [B, 1] combined mate
        """
        features = self.forward_backbone(layers, global_v)
        
        # Run all experts
        wdl_list = []
        mate_list = []
        for expert in self.experts:
            wdl, mate = expert(features)
            wdl_list.append(wdl)
            mate_list.append(mate)
        
        wdl_all = torch.stack(wdl_list, dim=1)   # [B, 4, 3]
        mate_all = torch.stack(mate_list, dim=1)  # [B, 4, 1]
        
        # Combine with weights
        if expert_weights is None:
            # Equal weighting
            combined_wdl = wdl_all.mean(dim=1)
            combined_mate = mate_all.mean(dim=1)
        else:
            # Weighted combination
            w = expert_weights.unsqueeze(-1)  # [B, 4, 1]
            combined_wdl = (wdl_all * w).sum(dim=1)  # [B, 3]
            combined_mate = (mate_all * w).sum(dim=1)  # [B, 1]
        
        return combined_wdl, combined_mate
    
    def forward_single_expert(self, layers, global_v, expert_idx=0):
        """Forward through backbone + single expert (for inference compatibility)."""
        features = self.forward_backbone(layers, global_v)
        return self.experts[expert_idx](features)
    
    def forward_top2(self, layers, global_v, expert_weights):
        """
        Efficient inference with top-2 expert routing.
        
        Only runs the 2 highest-weighted experts instead of all 4.
        This is the recommended inference method.
        
        Args:
            layers: [B, 32, 8, 8] board planes
            global_v: [B, n_globals] global features  
            expert_weights: [B, 4] expert weights (will be softmax normalized)
            
        Returns:
            wdl: [B, 3] combined WDL
            mate: [B, 1] combined mate
        """
        features = self.forward_backbone(layers, global_v)
        
        # Get top-2 experts by weight
        top2_values, top2_idx = torch.topk(expert_weights, k=2, dim=1)
        top2_contrib = top2_values / (top2_values.sum(dim=1, keepdim=True) + 1e-8)
        
        B = features.size(0)
        device = features.device
        
        pred_wdl = torch.zeros(B, 3, device=device)
        pred_mate = torch.zeros(B, 1, device=device)
        
        for k in range(2):  # Top-1 and Top-2
            for expert_idx in range(4):
                mask = (top2_idx[:, k] == expert_idx)
                if mask.sum() == 0:
                    continue
                wdl, mate = self.experts[expert_idx](features[mask])
                pred_wdl[mask] += wdl * top2_contrib[mask, k:k+1]
                pred_mate[mask] += mate * top2_contrib[mask, k:k+1]
        
        return pred_wdl, pred_mate
    
    def load_from_simple(self, simple_model):
        """
        Load backbone and expert[0] weights from a trained ChessNetWDL model.
        
        This allows skipping Phase 1 training entirely.
        
        Args:
            simple_model: Trained ChessNetWDL instance
        """
        # Copy backbone weights
        self.stem.load_state_dict(simple_model.stem.state_dict())
        self.bn_stem.load_state_dict(simple_model.bn_stem.state_dict())
        self.stem_global.load_state_dict(simple_model.stem_global.state_dict())
        self.bn_global.load_state_dict(simple_model.bn_global.state_dict())
        self.body.load_state_dict(simple_model.body.state_dict())
        
        # Copy head to expert[0]
        self.experts[0].head_conv.load_state_dict(simple_model.head_conv.state_dict())
        self.experts[0].head_hidden.load_state_dict(simple_model.head_hidden.state_dict())
        self.experts[0].head_wdl.load_state_dict(simple_model.head_wdl.state_dict())
        self.experts[0].head_mate.load_state_dict(simple_model.head_mate.state_dict())
        
        # Initialize other experts from expert[0] (fine-tune from pretrained)
        for i in range(1, 4):
            self.experts[i].load_state_dict(self.experts[0].state_dict())
        
        print(f"[LightMoE] Loaded backbone + experts from model_simple")
    
    def fuse_model(self):
        """Fuse BN for inference (RepVGG style)."""
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
        torch.quantization.fuse_modules(self, [['stem', 'bn_stem']], inplace=True)
        torch.quantization.fuse_modules(self, [['stem_global', 'bn_global']], inplace=True)


# =============================================================================
#   SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- ChessNetLightMoE Check on {device} ---")
    
    model = ChessNetLightMoE(n_filters=64, n_blocks=6).to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for n, p in model.named_parameters() 
                         if not n.startswith('experts'))
    expert_params = sum(p.numel() for n, p in model.named_parameters() 
                        if n.startswith('experts'))
    
    print(f"Total parameters: {total_params:,}")
    print(f"  Backbone: {backbone_params:,}")
    print(f"  Experts (4): {expert_params:,} ({expert_params//4:,} each)")
    
    # Test forward pass
    B = 2
    dummy_layers = torch.randn(B, 32, 8, 8).to(device)
    dummy_global = torch.randn(B, 15).to(device)
    dummy_weights = F.softmax(torch.randn(B, 4), dim=1).to(device)
    
    with torch.no_grad():
        # Test combined forward
        wdl, mate = model(dummy_layers, dummy_global, dummy_weights)
        print(f"\nCombined output shapes: wdl={wdl.shape}, mate={mate.shape}")
        print(f"WDL sample: {wdl[0].cpu().numpy()}")
        print(f"WDL sum: {wdl.sum(dim=1)}")
        
        # Test single expert forward
        wdl_e0, mate_e0 = model.forward_single_expert(dummy_layers, dummy_global, 0)
        print(f"\nExpert[0] shapes: wdl={wdl_e0.shape}, mate={mate_e0.shape}")
    
    print("\nStatus: OK ✓")
