import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
#   BUILDING BLOCKS
# =============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise attention."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) # Squeeze: (Batch, C, 8, 8) -> (Batch, C, 1, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True), # Compress
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True), # Expand
            nn.Sigmoid() # Importance Score (0.0 to 1.0)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # Scale the input features

class ResidualBlockSE(nn.Module):
    """ResNet Block with Squeeze-and-Excitation."""
    def __init__(self, channels, reduction=16):
        super(ResidualBlockSE, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction)

    def forward(self, x):    
        residual = x
        # Standard ResNet pass
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Squeeze-and-Excitation
        out = self.se(out)
        out += residual
        return F.relu(out)
    

class SelfAttentionBlock(nn.Module):
    """Spatial Self-Attention using Multi-Head Attention."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # MultiHeadAttention expects (seq, batch, dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=False  # -> (S,B,C)
        )

        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # Flatten: (B, C, H, W) -> (S=64, B, C) for MHA
        tokens_raw = x.view(B, C, -1).permute(2, 0, 1)

        # Pre-Norm architecture
        tokens_norm = self.ln(tokens_raw)

        # Self-Attention
        attn_out, _ = self.mha(tokens_norm, tokens_norm, tokens_norm)

        # Residual connection
        tokens = tokens_raw + attn_out

        # Reshape back to (B, C, H, W)
        out = tokens.permute(1, 2, 0).view(B, C, H, W)
        return out


# =============================================================================
#   HEADS
# =============================================================================
class SpatialValueHead(nn.Module):
    """Value Head that preserves spatial layout."""
    def __init__(self, channels, hidden=256):
        super().__init__()
        # 1x1 Conv to reduce depth before flattening (saves params)
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=1) 
        self.bn1 = nn.BatchNorm2d(32)
        
        # Flatten: 32 channels * 64 squares = 2048 features
        self.fc1 = nn.Linear(32 * 8 * 8, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.flatten(start_dim=1) # Preserves spatial layout!
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))
    


# =============================================================================
#   UTILITIES
# =============================================================================

def initialize_weights(m):
    """Kaiming Initialization for Conv layers."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# =============================================================================
#   MAIN MODEL (The "Assembler")
# =============================================================================

class ChessNet(nn.Module):
    def __init__(
        self, 
        n_inputs=42,            # Number of input planes
        n_filters=128,          # Width of the network
        n_pre_blocks=3,         # Residual blocks BEFORE attention
        n_post_blocks=2,        # Residual blocks AFTER attention
        use_attention=True,     # Toggle for the Self-Attention block
        init_weights=True       # Whether to initialize weights
    ):
        super(ChessNet, self).__init__()

        # 1. Stem
        self.stem = nn.Sequential(
            nn.Conv2d(n_inputs, n_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True)
        )

        # 2. Dynamic Body Construction
        blocks = []
        
        # A. Add Pre-Attention Residual Blocks
        for _ in range(n_pre_blocks):
            blocks.append(ResidualBlockSE(n_filters))
            
        # B. Add Optional Attention Block
        if use_attention:
            blocks.append(SelfAttentionBlock(n_filters))
            
        # C. Add Post-Attention Residual Blocks
        for _ in range(n_post_blocks):
            blocks.append(ResidualBlockSE(n_filters))

        # Register the list of layers as a Sequential module
        self.tower = nn.Sequential(*blocks)

        # 3. Value Head
        self.value_head = SpatialValueHead(n_filters)

        # 4. Initialize Weights
        if init_weights:
            self.apply(initialize_weights)

    def forward(self, board):
        x = self.stem(board)
        x = self.tower(x)
        return self.value_head(x)


# =============================================================================
#   SANITY CHECK: Run this block to verify the model works.
# =============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessNet().to(device)
    
    # Create a dummy batch: (Batch=2, Channels=42, Height=8, Width=8)
    dummy_input = torch.randn(2, 42, 8, 8).to(device)
    
    print(f"Model created on {device}")
    
    # Calculate parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {params:,}")
    
    # Test Forward Pass
    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}") # Should be (2, 1)
    print("Output Values:", output.detach().cpu().numpy())