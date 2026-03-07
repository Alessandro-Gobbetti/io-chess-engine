import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
#   BUILDING BLOCK: RepVGG Block
# =============================================================================
class RepVGGBlock(nn.Module):
    def __init__(self, channels):
        super(RepVGGBlock, self).__init__()
        self.is_deployed = False
        self.channels = channels
        
        # Training Branches
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn3x3 = nn.BatchNorm2d(channels)
        
        self.conv1x1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(channels)
        
        self.bn_identity = nn.BatchNorm2d(channels)
        
        # Inference Layer
        self.fused_conv = None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.is_deployed:
            return self.act(self.fused_conv(x))
        
        y = self.bn3x3(self.conv3x3(x))
        y += self.bn1x1(self.conv1x1(x))
        y += self.bn_identity(x)
        return self.act(y)

    def switch_to_deploy(self):
        if self.is_deployed: return
        
        k3, b3 = self._fuse_bn(self.conv3x3, self.bn3x3)
        k1, b1 = self._fuse_bn(self.conv1x1, self.bn1x1)
        k0, b0 = self._fuse_bn_identity(self.bn_identity)
        
        k_final = k3 + F.pad(k1, (1, 1, 1, 1)) + k0
        b_final = b3 + b1 + b0
        
        self.fused_conv = nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=True)
        self.fused_conv.weight.data = k_final
        self.fused_conv.bias.data = b_final
        
        del self.conv3x3, self.bn3x3, self.conv1x1, self.bn1x1, self.bn_identity
        self.is_deployed = True

    def _fuse_bn(self, conv, bn):
        w = conv.weight
        mean, var = bn.running_mean, bn.running_var
        gamma, beta = bn.weight, bn.bias
        eps = bn.eps
        std = (var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return w * t, beta - mean * gamma / std

    def _fuse_bn_identity(self, bn):
        k = torch.zeros(self.channels, self.channels, 3, 3, device=bn.weight.device)
        for i in range(self.channels):
            k[i, i, 1, 1] = 1.0
        mean, var = bn.running_mean, bn.running_var
        gamma, beta = bn.weight, bn.bias
        eps = bn.eps
        std = (var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return k * t, beta - mean * gamma / std

# =============================================================================
#   MAIN MODEL: ChessNet
# =============================================================================

class ChessNet(nn.Module):
    def __init__(self, n_filters=64, n_blocks=8):
        super().__init__()
        self.n_filters = n_filters
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.float_func = torch.nn.quantized.FloatFunctional()

        # --- UNIFIED STEM (32 Inputs) ---
        # Optimization: 30 Sparse + 2 Dense = 32 Channels (Multiple of 4/8/16)
        # This aligns perfectly with CPU/GPU registers.
        self.stem = nn.Conv2d(32, n_filters, 3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(n_filters)
        
        # Global Stem (14 Inputs)
        self.stem_global = nn.Linear(14, n_filters, bias=False) 
        self.bn_global   = nn.BatchNorm1d(n_filters) 

        self.stem_act = nn.ReLU(inplace=True)

        # Body
        self.body = nn.Sequential(*[RepVGGBlock(n_filters) for _ in range(n_blocks)])

        # Head
        self.head_conv = nn.Conv2d(n_filters, 32, 1)
        self.head_act = nn.ReLU(inplace=True)
        self.head_flat = nn.Flatten()
        self.head_lin1 = nn.Linear(32 * 64, 128)
        self.head_act2 = nn.ReLU(inplace=True)
        self.head_lin2 = nn.Linear(128, 1)
        
        # Output Activation: range [-1, 1]
        self.head_tanh = nn.Tanh()

    def forward(self, layers, global_v):
        x = self.quant(layers)
        g_in = self.quant(global_v)

        # Process Board
        x = self.bn_stem(self.stem(x))
        
        # Process Global
        x_g = self.bn_global(self.stem_global(g_in))

        # PRINT THE RANGES
        # if self.training and torch.rand(1).item() < 0.01: # Check rarely
        #     print(f"Board Range: {x.min().item():.3f} to {x.max().item():.3f}")
        #     print(f"Global Range: {x_g.min().item():.3f} to {x_g.max().item():.3f}")
        # Broadcast context: [B, C] -> [B, C, 1, 1]
        x_g = x_g.view(-1, self.n_filters, 1, 1) 

        # Add Global Context
        x = self.float_func.add(x, x_g)
        x = self.stem_act(x)

        # Body
        x = self.body(x)
        
        # Head
        x = self.head_conv(x)
        x = self.head_act(x)
        x = self.dequant(x)
        x = self.head_flat(x)
        x = self.head_lin1(x)
        x = self.head_act2(x)
        x = self.head_lin2(x)
        x = self.head_tanh(x) # Squashing

        return x

    def fuse_model(self):
        # Collapse RepVGG branches
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()

        # Fuse standard Conv+BN layers (stem and global)
        # RepVGG blocks are already fused at this point
        torch.quantization.fuse_modules(self, [['stem', 'bn_stem']], inplace=True)
        torch.quantization.fuse_modules(self, [['stem_global', 'bn_global']], inplace=True)

# =============================================================================
#   SANITY CHECK
# =============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- ChessNet Check on {device} ---")
    
    model = ChessNet(n_filters=64, n_blocks=4).to(device)
    model.eval()

    # Input Check: 30 Sparse + 2 Dense = 32 Total
    B = 2
    dummy_layers = torch.randn(B, 32, 8, 8).to(device) 
    dummy_global = torch.randn(B, 14).to(device) # 14 Globals

    print(f"Input Shape: {dummy_layers.shape[1]} channels")

    with torch.no_grad():
        out = model(dummy_layers, dummy_global)
    
    print(f"Output Shape: {out.shape}")
    print(f"Output Value Range: [{out.min():.4f}, {out.max():.4f}]")
    print("Status: OK")
