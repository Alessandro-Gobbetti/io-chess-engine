import torch
import torch.nn as nn
from time import perf_counter
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import wandb
from utils import Colors
import sys
import os


def train_model(model, train_loader, val_loader, config, experiment_name, device='auto'):
    
    epochs = config['epochs']
    lr = config['lr']

    # 1. OPTIMIZATION: Enable Algo Finder
    torch.backends.cudnn.benchmark = True

    # 1. Setup Device (CUDA / MPS / CPU)
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Using Device: {Colors.GREEN}CUDA (NVIDIA){Colors.RESET}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"🚀 Using Device: {Colors.GREEN}MPS (Apple Silicon){Colors.RESET}")
        else:
            device = torch.device('cpu')
            print(f"⚠️  Using Device: {Colors.YELLOW}CPU{Colors.RESET} (This might be slow)")
    else:
        device = torch.device(device)
        print(f"🚀 Using Device: {Colors.GREEN}{device}{Colors.RESET}")


    device_type = device.type if device.type != 'mps' else 'cpu'  # AMP only supports 'cuda' or 'cpu'


    # Move model to GPU
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # --- AMP SETUP ---
    # Scaler handles the dynamic range of float16 gradients to prevent underflow
    # scaler = GradScaler(device=device_type)


    best_loss = float('inf')
    
    # --- GLOBAL STEP ---
    # Keeps W&B x-axis continuous across epochs
    global_step = 0 

    # Header
    print(f"\n{Colors.BOLD}{'='*85}{Colors.RESET}")
    print(f"{Colors.BOLD}{'CHESSNET TRAINING DASHBOARD':^85}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*85}{Colors.RESET}")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^12} | {'Delta':^10} | {'Time':^8} | {'Status':^10}")
    print(f"{'-'*85}")

    try:
        for epoch in range(epochs):
            epoch_start = perf_counter()
            model.train()
            total_train_loss = 0.0
            
            # --- FIX 1: file=sys.stdout ---
            # Forces tqdm to use the same stream as print(), preventing overwritten history.
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                        unit="batch", leave=False, ncols=100, file=sys.stdout)
            
            for batch_idx, (board, target) in enumerate(pbar):
                board = board.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True).unsqueeze(1)

                # 2. Cast to Float on GPU (Required for Conv2d)
                board = board.to(dtype=torch.float32).div_(255.0)
                target = target.to(dtype=torch.float32) # Targets usually need float for MSELoss too

                optimizer.zero_grad()

                # Forward
                output = model(board)
                loss = criterion(output, target)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                # Update metrics
                current_loss = loss.item()
                total_train_loss += current_loss
                
                # Update the progress bar with current loss (smooth average)
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
                
                # Log batch loss to wandb
                global_step += 1
                if global_step % 10 == 0:
                    wandb.log({"batch_train_loss": current_loss}, step=global_step)

            avg_train_loss = total_train_loss / len(train_loader)

            # --- Validation Loop (Fast, no gradient) ---
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                # Also use sys.stdout for validation bar
                for board, target in tqdm(val_loader, desc="Validating", unit="batch", leave=False, ncols=100, file=sys.stdout):
                    board = board.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True).unsqueeze(1)

                    # 2. Cast to Float on GPU (Required for Conv2d)
                    board = board.to(dtype=torch.float32).div_(255.0)
                    target = target.to(dtype=torch.float32) 

                    output = model(board)
                    loss = criterion(output, target)

                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            
            # Scheduler step (reduce LR if stuck)
            scheduler.step(avg_val_loss)

            # Log epoch metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "loss/train": avg_train_loss,
                "loss/val": avg_val_loss,
                "loss/delta": avg_val_loss - avg_train_loss,
                "lr": optimizer.param_groups[0]['lr']
            }, step=global_step)

            # --- Checkpointing & Formatting ---
            duration = perf_counter() - epoch_start
            delta = avg_val_loss - avg_train_loss
            
            is_best = avg_val_loss < best_loss
            if is_best:
                best_loss = avg_val_loss
                save_path = os.path.join("models", f"{experiment_name}_best.pth")
                torch.save(model.state_dict(), save_path)
                status_str = f"{Colors.GREEN}NEW BEST{Colors.RESET}"
            else:
                status_str = f"{Colors.YELLOW}-{Colors.RESET}"

            # Print the clean history row
            print(f"{epoch+1:^7d} | {avg_train_loss:^12.8f} | {avg_val_loss:^12.8f} | {delta:^10.4f} | {duration:^7.1f}s | {status_str}")
            
            # Force flush to ensure it appears in terminal immediately
            sys.stdout.flush()

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Training interrupted by user.{Colors.RESET}")
    
    # Save final
    print(f"{'-'*85}")
    final_path = os.path.join("models", f"{experiment_name}_final.pth")
    print(f"Saving final model to {Colors.CYAN}{final_path}{Colors.RESET}...")
    torch.save(model.state_dict(), final_path)
    print("Done.")