"""
@file prepare_dataset.py
@brief Downloads and prepares the chess dataset from HuggingFace, filtering by depth and formatting evaluations.
"""
import os
import csv
import matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm # Clean, single-line progress bar

def main():
    print("Downloading/Loading dataset locally (Memory-Mapped)...")
    dataset = load_dataset("mateuszgrzyb/lichess-stockfish-normalized", split="train")
    total_rows = len(dataset)
    print(f"Dataset loaded. Total rows: {total_rows:,}")

    output_csv = "chess_dataset.csv"
    
    # We will keep track of depth counts manually as we loop!
    depth_counts = Counter()
    batch_size = 100_000

    print(f"Processing data and writing to {output_csv} in chunks...")
    print("(This uses minimal RAM because it only holds 100k rows in memory at a time)")

    # Open the CSV file directly using Python's standard CSV writer
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "eval"]) # Write the header
        
        # dataset.iter() loops through the memory-mapped file in tiny, safe chunks
        for batch in tqdm(dataset.iter(batch_size=batch_size), total=(total_rows // batch_size) + 1, desc="Processing"):
            
            rows_to_write = []
            
            # Extract columns for this specific batch
            fens = batch["fen"]
            mates = batch["mate"]
            cps = batch["cp"]
            depths = batch["depth"]
            
            for fen, m, cp, d in zip(fens, mates, cps, depths):
                # Skip lines with depth <= 10
                if d is not None and d <= 10:
                    continue

                # 1. Format the evaluation string
                if m is not None:
                     eval_str = f"#{m}"
                else:
                     eval_str = str(cp)
                
                rows_to_write.append([fen, eval_str])
                
                # 2. Count the depth for our histogram
                if d is not None:
                    depth_counts[d] += 1
                    
            # 3. Write the batch of 100,000 directly to disk and flush it from RAM
            writer.writerows(rows_to_write)
            
    print(f"\nFinished generating {output_csv}!")

    # --- Plotting the Histogram ---
    if depth_counts:
        print("Generating depths histogram...")
        plt.figure(figsize=(10, 6))
        
        # Sort the depths for plotting
        depths_keys = sorted(depth_counts.keys())
        frequencies = [depth_counts[d] for d in depths_keys]
        
        # Plot using plt.bar
        plt.bar(depths_keys, frequencies, edgecolor='black', alpha=0.7, width=1.0)
        
        plt.title('Distribution of Evaluation Depths', fontsize=14)
        plt.xlabel('Depth', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        plt.xticks(range(min(depths_keys), max(depths_keys) + 1, max(1, len(depths_keys)//15)))
        plt.grid(axis='y', alpha=0.75)
        
        histogram_filename = "depth_histogram.png"
        plt.savefig(histogram_filename)
        print(f"Histogram successfully saved as {histogram_filename}!")
    else:
        print("No depth data found to create a histogram.")

if __name__ == "__main__":
    main()