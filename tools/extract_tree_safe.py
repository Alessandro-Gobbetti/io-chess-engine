#!/usr/bin/env python3
import sys
import subprocess
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run chess engine and extract search tree safely")
    parser.add_argument("--engine", required=True, help="Path to engine binary")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--fen", default="startpos", help="FEN string")
    parser.add_argument("--depth", type=int, default=10, help="Search depth")
    parser.add_argument("--export-depth", type=int, default=4, help="Tree export depth")
    args = parser.parse_args()

    cmd = [args.engine, "--model", args.model]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr, # Pass stderr through
            text=True,
            bufsize=1 # Line buffered
        )
    except Exception as e:
        print(f"Error starting engine: {e}", file=sys.stderr)
        sys.exit(1)

    # Helper to send commands
    def send(msg):
        try:
            process.stdin.write(msg + "\n")
            process.stdin.flush()
        except BrokenPipeError:
            print("Engine closed connection unexpectedly", file=sys.stderr)
            sys.exit(1)

    # Build UCI commands
    commands = [
        "uci",
        "setoption name EvalDevice value ONNX-CPU",
        "setoption name Threads value 1",
        "setoption name EvalThreads value 2",
        "setoption name ExportTree value true",
        f"setoption name ExportTreeDepth value {args.export_depth}",
        "ucinewgame"
    ]

    if args.fen == "startpos":
        commands.append("position startpos")
    else:
        commands.append(f"position fen {args.fen}")

    commands.append(f"go depth {args.depth}")

    # Send initialization and go
    for c in commands:
        send(c)
    
    # Read output loop
    json_found = False
    
    while True:
        line = process.stdout.readline()
        if not line:
            break
        
        # Print line to stdout so extract_tree.sh can capture it
        print(line, end="")
        
        # Check for completion
        if line.startswith("bestmove"):
            # Search finished!
            send("quit")
            break
            
    # Wait for exit
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

if __name__ == "__main__":
    main()
