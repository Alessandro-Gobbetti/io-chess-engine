"""
@file model.py
@brief Lightweight MoE Model for Chess Evaluation (Factorized Cache Architecture).

RULES FOR C++ PERFORMANCE:
1. NO 3x3 CONVOLUTIONS AFTER THE MIXER. (Preserves spatial caching).
2. Dense layers must be protected by a 1x1 bottleneck to minimize Flat size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PieceBranch(nn.Module):
    """
    The deep, cacheable spatial logic for a single piece type.
    Because these are computed BEFORE the mixer, they only cost CPU cycles
    when the specific piece type moves.
    """
    def __init__(self, in_channels, mid_channels=16):
        super().__init__()
        # Layer 0: 3x3 Conv
        self.conv0 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        
        # Layer 1: depthwise 3x3 Conv (groups=channels)
        self.conv1 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            groups=mid_channels,
        )
        
        # Layer 2: pointwise 1x1 Conv
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.conv0(x))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class LightExpert(nn.Module):
    """
    The ultra-fast Dense Expert. 
    It compresses the massive Mixer output down to a small bottleneck before flattening.
    """
    def __init__(
        self,
        mixer_channels=64,
        bottleneck_channels=32,
        hidden_dim=128,
        expert_pool="flat",
    ):
        super().__init__()
        if expert_pool not in {"flat", "gap", "pool2avg", "pool2max"}:
            raise ValueError(
                "expert_pool must be one of: flat, gap, pool2avg, pool2max"
            )
        self.expert_pool = expert_pool
        
        # 1x1 Bottleneck Compressor (e.g., 64 -> 32 channels)
        self.head_conv = nn.Conv2d(mixer_channels, bottleneck_channels, kernel_size=1)
        self.head_act = nn.ReLU(inplace=True)

        if expert_pool == "flat":
            hidden_in = bottleneck_channels * 64
        elif expert_pool == "gap":
            hidden_in = bottleneck_channels
        else:
            hidden_in = bottleneck_channels * 16

        # Fast Dense Layer after chosen pooling path.
        self.head_hidden = nn.Linear(hidden_in, hidden_dim)
        self.head_act2 = nn.ReLU(inplace=True)
        
        # Outputs
        self.head_wdl = nn.Linear(hidden_dim, 3)
    
    def forward(self, x):
        # Compress & ReLU
        x = self.head_act(self.head_conv(x))

        if self.expert_pool == "flat":
            x = torch.flatten(x, start_dim=1)
        elif self.expert_pool == "gap":
            x = x.mean(dim=(2, 3))
        elif self.expert_pool == "pool2avg":
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            x = torch.flatten(x, start_dim=1)
        else:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = torch.flatten(x, start_dim=1)

        # Dense Math
        x = self.head_act2(self.head_hidden(x))

        # Per-expert softmax WDL.
        wdl = F.softmax(self.head_wdl(x), dim=1)
        return wdl


class ChessNetFactorizedMoE(nn.Module):
    # Channel counts per branch (Rich setup):
    # Pawns/Kings/Knights=4, Sliders=5 (x-ray).
    # Order: WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK
    PLANES_PER_TYPE = [4, 4, 5, 5, 5, 4, 4, 4, 5, 5, 5, 4]

    def __init__(self, 
                 n_globals=21, 
                 branch_dim=16, 
                 mixer_out=64, 
                 n_bypass=12, 
                 n_experts=4,
                 expert_bottleneck=32,
                 expert_hidden=128,
                 expert_pool="flat"):
        super().__init__()
        if expert_pool not in {"flat", "gap", "pool2avg", "pool2max"}:
            raise ValueError(
                "expert_pool must be one of: flat, gap, pool2avg, pool2max"
            )
        self.expert_pool = expert_pool
        self.n_experts = n_experts
        
        # 1. The 12 Independent Piece Branches
        self.branches = nn.ModuleList([
            PieceBranch(in_channels=in_ch, mid_channels=branch_dim) 
            for in_ch in self.PLANES_PER_TYPE
        ])
        
        # 2. The Pointwise Mixer (Delta-Accumulator target)
        mixer_in_channels = (12 * branch_dim) + n_bypass
        self.pointwise_mixer = nn.Conv2d(mixer_in_channels, mixer_out, kernel_size=1)
        
        # 3. Global Scalar Injection
        self.stem_global = nn.Linear(n_globals, mixer_out)
        
        self.mixer_act = nn.ReLU(inplace=True)
        
        # 4. The Experts
        self.experts = nn.ModuleList([
            LightExpert(
                mixer_channels=mixer_out, 
                bottleneck_channels=expert_bottleneck, 
                hidden_dim=expert_hidden,
                expert_pool=expert_pool,
            ) for _ in range(n_experts)
        ])

    def forward(self, *inputs, weights=None):
        if len(inputs) == 3 and isinstance(inputs[0], (list, tuple)):
            planes_list, bypass, global_v = inputs
        elif len(inputs) == 14:
            planes_list = list(inputs[:12])
            bypass = inputs[12]
            global_v = inputs[13]
        else:
            raise ValueError(
                "Expected either (planes_list, bypass, global_v) or 14 tensor inputs "
                "(12 piece planes, bypass, global_v)."
            )

        # 1. Evaluate Branches (In C++, 90% of these are skipped and fetched from cache)
        branch_outs = []
        for i, branch in enumerate(self.branches):
            branch_outs.append(branch(planes_list[i]))
            
        # 2. Concatenate Branches + Bypass Lanes
        x = torch.cat(branch_outs + [bypass], dim=1)
        
        # 3. Pointwise Mixer
        x = self.pointwise_mixer(x)
        
        # 4. Inject Globals and apply ReLU
        g = self.stem_global(global_v).view(-1, x.shape[1], 1, 1)
        x = self.mixer_act(x + g)

        # 5) Evaluate all experts.
        expert_wdl = []
        for expert in self.experts:
            w = expert(x)
            expert_wdl.append(w)
        expert_wdl = torch.stack(expert_wdl, dim=1)  # [B, E, 3]

        # 6) Mixture routing: callers should provide expert weights.
        # If omitted (e.g. simple smoke tests), fall back to uniform mixing.
        w = weights
        if w is None:
            w = torch.full(
                (expert_wdl.size(0), self.n_experts),
                1.0 / float(self.n_experts),
                dtype=expert_wdl.dtype,
                device=expert_wdl.device,
            )
        if w.dim() == 1:
            w = w.unsqueeze(0)
        if w.size(1) != self.n_experts:
            raise ValueError(
                f"weights must have shape [B, {self.n_experts}] or [{self.n_experts}]"
            )
        wdl_out = (expert_wdl * w.unsqueeze(-1)).sum(dim=1)
            
        return wdl_out
    



if __name__ == "__main__":
    # Quick test to verify dimensions and forward pass
    model = ChessNetFactorizedMoE(expert_bottleneck=16, mixer_out=512)
    model.eval()
    
    # Dummy input data
    planes_list = [torch.randn(1, in_ch, 8, 8) for in_ch in ChessNetFactorizedMoE.PLANES_PER_TYPE]
    bypass = torch.randn(1, 12, 8, 8)
    global_v = torch.randn(1, 21)


    # print number of parameters    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
    wdl = model(planes_list, bypass, global_v)
    print("WDL Output Shape:", wdl.shape)  # Expected: (1, 3)

    # printing number of parameters in each part
    print("\nParameter Counts:")
    total_params = sum(p.numel() for p in model.parameters())
    branches_params = sum(p.numel() for n, p in model.named_parameters() if "branches" in n)
    stem_global_params = sum(p.numel() for n, p in model.named_parameters() if "stem_global" in n)
    pointwise_mixer_params = sum(p.numel() for n, p in model.named_parameters() if "pointwise_mixer" in n)
    backbone_params = sum(p.numel() for n, p in model.named_parameters() if "branches" in n or "pointwise_mixer" in n or "stem_global" in n)
    expert_params = sum(p.numel() for n, p in model.named_parameters() if "experts" in n)
    print(f"  Total: {total_params}")
    print(f"  Branches: {branches_params} ({branches_params/total_params:.2%})")
    print(f"  Stem Global: {stem_global_params} ({stem_global_params/total_params:.2%})")
    print(f"  Pointwise Mixer: {pointwise_mixer_params} ({pointwise_mixer_params/total_params:.2%})")
    print(f"  Backbone (Branches + Mixer + Stem): {backbone_params} ({backbone_params/total_params:.2%})")
    print(f"  Experts: {expert_params} ({expert_params/total_params:.2%})")
