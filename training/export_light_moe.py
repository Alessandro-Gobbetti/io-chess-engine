#!/usr/bin/env python3
"""
Export ChessNetLightMoE as separate ONNX models for efficient top-2 inference.

Exports:
  - backbone.onnx: Shared feature extractor
  - expert0.onnx, expert1.onnx, expert2.onnx, expert3.onnx: Individual expert heads

C++ inference flow:
  1. Compute expert weights (via C++ ExpertRouter)
  2. Determine top-2 expert indices
  3. Run backbone.onnx once → features
  4. Run only top-2 expert ONNXes on features
  5. Combine outputs with normalized weights

This approach runs only 2 experts instead of 4, saving ~50% inference time.

Usage:
    python export_light_moe.py --checkpoint checkpoints/light_moe_best.pt --output_dir ./onnx/
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_light_moe import ChessNetLightMoE


class BackboneForExport(nn.Module):
    """
    Backbone-only model for ONNX export.
    
    Input:
        layers: [B, 32, 8, 8] float32 board planes (0-255 range)
        global_features: [B, 15] float32 global features
        
    Output:
        features: [B, n_filters, 8, 8] spatial features for experts
    """
    def __init__(self, model: ChessNetLightMoE):
        super().__init__()
        self.stem = model.stem
        self.bn_stem = model.bn_stem
        self.stem_global = model.stem_global
        self.bn_global = model.bn_global
        self.stem_act = model.stem_act
        self.body = model.body
        self.n_filters = model.n_filters
    
    def forward(self, layers: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layers: [B, 32, 8, 8] uint8 board planes (0-255 range)
            global_features: [B, 15] float32 global features
            
        Returns:
            features: [B, n_filters, 8, 8] spatial features
        """
        # Normalize inputs (0-255 -> 0-1) embedded in graph
        # Input is uint8, cast to float first
        x = self.bn_stem(self.stem(layers.to(torch.float32) / 255.0))
        x_g = self.bn_global(self.stem_global(global_features))
        x_g = x_g.view(-1, self.n_filters, 1, 1)
        x = self.stem_act(x + x_g)
        x = self.body(x)
        return x

class ExpertForExport(nn.Module):
    """
    Single expert head for ONNX export.
    
    Input:
        features: [B, n_filters, 8, 8] spatial features from backbone
        
    Output:
        wdl_mate: [B, 4] tensor of [win, draw, loss, mate_dist]
    """
    def __init__(self, expert: nn.Module):
        super().__init__()
        self.head_conv = expert.head_conv
        self.head_act = expert.head_act
        self.head_flat = expert.head_flat
        self.head_hidden = expert.head_hidden
        self.head_act2 = expert.head_act2
        self.head_wdl = expert.head_wdl
        self.head_mate = expert.head_mate
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, n_filters, 8, 8] spatial features from backbone
            
        Returns:
            wdl_mate: [B, 4] tensor of [win, draw, loss, mate_dist]
        """
        x = self.head_act(self.head_conv(features))
        x = self.head_flat(x)
        x = self.head_act2(self.head_hidden(x))
        
        wdl = F.softmax(self.head_wdl(x), dim=1)  # [B, 3]
        mate = torch.sigmoid(self.head_mate(x))    # [B, 1]
        
        return torch.cat([wdl, mate], dim=1)  # [B, 4]


def simplify_onnx(path: str):
    """Simplify ONNX model using onnx-simplifier if available."""
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("  ⚠ onnx-simplifier not installed (skipping simplification)")
        return

    try:
        print(f"  Simplifying {os.path.basename(path)}...")
        model = onnx.load(path)
        model_simp, check = simplify(model)
        if check:
            onnx.save(model_simp, path)
            print(f"  ✓ Simplified! New size: {os.path.getsize(path)/1024:.1f} KB")
        else:
            print("  ⚠ Simplification check failed")
    except Exception as e:
        print(f"  ⚠ Simplification error: {e}")


def export_backbone(model: ChessNetLightMoE, output_path: str, n_filters: int = 64):
    """Export backbone to ONNX."""
    backbone = BackboneForExport(model)
    backbone.eval()
    
    # Dummy inputs
    B = 1
    # Inputs: layers should be uint8 for export
    dummy_layers = torch.randint(0, 256, (B, 32, 8, 8), dtype=torch.uint8)
    dummy_global = torch.randn(B, 15)
    
    # Test forward
    with torch.no_grad():
        features = backbone(dummy_layers, dummy_global)
        print(f"  Backbone output shape: {features.shape}")
    
    # Export
    torch.onnx.export(
        backbone,
        (dummy_layers, dummy_global),
        output_path,
        input_names=['layers', 'global_features'],
        output_names=['features'],
        dynamic_axes={
            'layers': {0: 'batch'},
            'global_features': {0: 'batch'},
            'features': {0: 'batch'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"  ✓ Exported backbone: {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    simplify_onnx(output_path)


def export_expert(model: ChessNetLightMoE, expert_idx: int, output_path: str, n_filters: int = 64):
    """Export single expert to ONNX."""
    expert = ExpertForExport(model.experts[expert_idx])
    expert.eval()
    
    # Dummy input (features from backbone)
    B = 1
    dummy_features = torch.randn(B, n_filters, 8, 8)
    
    # Test forward
    with torch.no_grad():
        output = expert(dummy_features)
        print(f"  Expert {expert_idx} output shape: {output.shape}")
    
    # Export
    torch.onnx.export(
        expert,
        (dummy_features,),
        output_path,
        input_names=['features'],
        output_names=['wdl_mate'],
        dynamic_axes={
            'features': {0: 'batch'},
            'wdl_mate': {0: 'batch'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"  ✓ Exported expert {expert_idx}: {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    simplify_onnx(output_path)


def verify_exports(output_dir: str, n_filters: int = 64):
    """Verify exported models with ONNX Runtime."""
    try:
        import onnxruntime as ort
        import numpy as np
        
        print("\nVerifying with ONNX Runtime...")
        
        # Load models
        backbone_session = ort.InferenceSession(os.path.join(output_dir, "backbone.onnx"))
        expert_sessions = [
            ort.InferenceSession(os.path.join(output_dir, f"expert{i}.onnx"))
            for i in range(4)
        ]
        
        # Create inputs
        B = 2
        # Match model input type (uint8)
        layers = np.random.randint(0, 256, (B, 32, 8, 8)).astype(np.uint8)
        global_features = np.random.randn(B, 15).astype(np.float32)
        expert_weights = np.array([[0.5, 0.3, 0.1, 0.1], [0.2, 0.2, 0.4, 0.2]], dtype=np.float32)
        
        # Run backbone
        features = backbone_session.run(None, {
            'layers': layers,
            'global_features': global_features,
        })[0]
        print(f"  Backbone output: {features.shape}")
        
        # Compute top-2
        top2_idx = np.argsort(expert_weights, axis=1)[:, -2:][:, ::-1]  # [B, 2]
        top2_weights = np.take_along_axis(expert_weights, top2_idx, axis=1)
        top2_weights = top2_weights / top2_weights.sum(axis=1, keepdims=True)
        
        print(f"  Top-2 indices: {top2_idx}")
        print(f"  Top-2 weights: {top2_weights}")
        
        # Run only top-2 experts
        final_output = np.zeros((B, 4), dtype=np.float32)
        
        for b in range(B):
            for k in range(2):
                exp_idx = top2_idx[b, k]
                weight = top2_weights[b, k]
                
                # Run single expert
                exp_output = expert_sessions[exp_idx].run(None, {
                    'features': features[b:b+1],
                })[0]
                
                final_output[b] += exp_output[0] * weight
        
        print(f"\n  ✓ Verification passed!")
        print(f"  Final output shape: {final_output.shape}")
        print(f"  Sample output: {final_output[0]}")
        print(f"  WDL sum: {final_output[0, :3].sum():.4f} (should be ~1.0)")
        
    except ImportError:
        print("\n⚠ onnxruntime not installed, skipping verification")

def export_all(checkpoint_path: str, output_dir: str, n_filters: int = 64, n_blocks: int = 6):
    """Export backbone + all 4 experts to separate ONNX files."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    device = torch.device('cpu')
    model = ChessNetLightMoE(n_filters=n_filters, n_blocks=n_blocks)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    # FUSE BN layers for faster inference
    print("Fusing BN layers (RepVGG)...")
    model.eval()  # Must be in eval mode for fusion
    model.fuse_model()
    
    print(f"Loaded from phase {ckpt.get('phase', '?')}, val_loss={ckpt.get('val_loss', 0):.5f}")
    print(f"\nExporting to {output_dir}/...")
    
    # Export backbone
    export_backbone(model, os.path.join(output_dir, "backbone.onnx"), n_filters)
    
    # Export each expert
    expert_names = ['tactical', 'strategic', 'major_end', 'minor_end']
    for i in range(4):
        export_expert(model, i, os.path.join(output_dir, f"expert{i}.onnx"), n_filters)
    
    # Summary
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir) if f.endswith('.onnx')
    )
    
    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}")
    print(f"  Files:")
    print(f"    - backbone.onnx (shared, run once)")
    print(f"    - expert0.onnx (Tactical)")
    print(f"    - expert1.onnx (Strategic)")
    print(f"    - expert2.onnx (Major Endgame)")
    print(f"    - expert3.onnx (Minor Endgame)")
    print(f"  Total size: {total_size/1024/1024:.2f} MB")
    print()
    print(f"C++ inference flow:")
    print(f"  1. Compute expert weights (ExpertRouter)")
    print(f"  2. Get top-2 indices and normalize weights")
    print(f"  3. Run backbone.onnx → features")
    print(f"  4. Run ONLY top-2 expert ONNXes on features")
    print(f"  5. Weighted average of outputs")
    
    # Verify
    verify_exports(output_dir, n_filters)


def main():
    parser = argparse.ArgumentParser(description="Export LightMoE to separate ONNXes")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="./onnx",
                        help="Output directory for ONNX files")
    parser.add_argument("--n_filters", type=int, default=64)
    parser.add_argument("--n_blocks", type=int, default=6)
    args = parser.parse_args()
    
    export_all(args.checkpoint, args.output_dir, args.n_filters, args.n_blocks)


if __name__ == "__main__":
    main()
