import torch
from torch.utils.data import DataLoader, random_split
import wandb
import argparse
import os
from dataset import ChessDataset
from model import ChessNet
from train import train_model
from utils import Colors, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train ChessNet")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--workers", type=int, default=6, help="Number of data loader workers")
    
    # --- Data Paths ---
    parser.add_argument("--features", type=str, default="../data/processed/features.bin", help="Path to features file")
    parser.add_argument("--labels", type=str, default="../data/processed/labels.bin", help="Path to labels file")

    # --- Model Architecture Settings ---
    parser.add_argument("--filters", type=int, default=128, help="Width of the network (n_filters)")
    parser.add_argument("--pre_blocks", type=int, default=3, help="Residual blocks BEFORE attention")
    parser.add_argument("--post_blocks", type=int, default=2, help="Residual blocks AFTER attention")
    parser.add_argument("--no_attention", action="store_false", dest="use_attention", help="Disable Self-Attention block")
    parser.set_defaults(use_attention=True)

    return parser.parse_args()

def main():
    set_seed(42)
    args = parse_args()

    # --- Configuration ---
    config = vars(args)
    config["architecture"] = "ChessNet"

    # 1. Initialize Model (Moved up to get param count)
    print(f"Initializing ChessNet (Filters={config['filters']}, Pre={config['pre_blocks']}, Post={config['post_blocks']}, Attn={config['use_attention']})...")
    model = ChessNet(
        n_filters=config['filters'],
        n_pre_blocks=config['pre_blocks'],
        n_post_blocks=config['post_blocks'],
        use_attention=config['use_attention']
    )
    
    # Calculate params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {Colors.CYAN}{params:,}{Colors.RESET}")
    config['trainable_parameters'] = params

    # Generate Experiment Name
    attn_str = "Attn" if config['use_attention'] else "NoAttn"
    if params > 1_000_000:
        param_str = f"{params/1_000_000:.1f}M"
    else:
        param_str = f"{params/1_000:.0f}K"
    experiment_name = f"ChessNet_{config['filters']}f_{config['pre_blocks']}pre_{config['post_blocks']}post_{attn_str}_{param_str}"

    # --- WandB Init ---
    wandb.init(
        project="io-chess-engine",
        name=experiment_name,
        config=config
    )

    # define metrics
    wandb.define_metric("global_step")
    wandb.define_metric("batch_train_loss", step_metric="global_step")
    wandb.define_metric("learning_rate", step_metric="global_step")

    wandb.define_metric("epoch")
    wandb.define_metric("avg_train_loss", step_metric="epoch")
    wandb.define_metric("avg_val_loss", step_metric="epoch")
    wandb.define_metric("learning_rate", step_metric="epoch")


    print(f"{Colors.BOLD}--- Starting Experiment: {experiment_name} ---{Colors.RESET}")

    # 1. Load Dataset
    print(f"Loading dataset from {config['features']}...")
    full_dataset = ChessDataset(config['features'], config['labels'])
    
    # 2. Split Train/Val
    total_size = len(full_dataset)
    val_size = int(total_size * config['val_split'])
    train_size = total_size - val_size
    
    split_generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=split_generator)
    print(f"Data loaded: {total_size} samples ({train_size} train, {val_size} val)")

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        pin_memory=True, 
        num_workers=config['workers'],
        persistent_workers=True,
        prefetch_factor=40
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        num_workers=config['workers'], 
        persistent_workers=True,
        prefetch_factor=4
    )


    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # 5. Start Training
    train_model(model, train_loader, val_loader, config, experiment_name)

if __name__ == "__main__":
    main()