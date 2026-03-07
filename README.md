<div align="center">

# io-chess

<a href="https://lichess.org/@/io-bot"><img src="https://img.shields.io/badge/Lichess-@io--bot-white?style=for-the-badge&logo=lichess&logoColor=white" style="border-radius: 4px;" alt="Lichess Bot"></a>

<a href="https://lichess.org/@/io-bot/perf/bullet"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2Fio-bot&query=%24.perfs.bullet.rating&label=Bullet&color=D55E00&style=for-the-badge&logo=lichess&logoColor=white" style="border-radius: 4px;height: 24px;" alt="Bullet Rating"></a>
<a href="https://lichess.org/@/io-bot/perf/blitz"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2Fio-bot&query=%24.perfs.blitz.rating&label=Blitz&color=E69F00&style=for-the-badge&logo=lichess&logoColor=white" style="border-radius: 4px;height: 24px;" alt="Blitz Rating"></a>
<a href="https://lichess.org/@/io-bot/perf/rapid"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2Fio-bot&query=%24.perfs.rapid.rating&label=Rapid&color=009E73&style=for-the-badge&logo=lichess&logoColor=white" style="border-radius: 4px;height: 24px;" alt="Rapid Rating"></a>
<br>
<a href="https://lichess.org/@/io-bot"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2Fio-bot&query=%24.count.all&label=Total%20Games&color=0072B2&style=for-the-badge" style="border-radius: 4px;height: 24px;" alt="Total Games"></a>
<a href="https://lichess.org/@/io-bot"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2Fio-bot&query=%24.count.win&label=Wins&color=009E73&style=for-the-badge" style="border-radius: 4px;height: 24px;" alt="Wins"></a>
<a href="https://lichess.org/@/io-bot"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Flichess.org%2Fapi%2Fuser%2Fio-bot&query=%24.count.loss&label=Losses&color=D55E00&style=for-the-badge" style="border-radius: 4px;height: 24px;" alt="Losses"></a>

<br />

[**Challenge on Lichess**](https://lichess.org/@/io-bot) &nbsp;&nbsp;•&nbsp;&nbsp; [**Web Engine Showcase**](#)

</div>

<br/>

**io-chess** is a custom UCI-compatible chess engine written in C++20. The evaluation relies on an ONNX-backed Mixture of Experts (MoE) neural network alongside a custom alpha-beta search implementation. The engine compiles natively for Linux/macOS and provides a WebAssembly target for browser execution.

## Architecture Overview

### Search
The search function uses a Principal Variation Search (PVS) with Iterative Deepening. Threading is handled via Lazy SMP for parallel node evaluation. Standard search techniques include:
*   **Extensions**: Singular Extensions, Check Extensions.
*   **Reductions & Pruning**: Null Move Pruning (NMP), Late Move Reductions (LMR), ProbCut, Futility Pruning, and Reverse Futility Pruning.
*   **Move Ordering**: Hash move priority, MVV-LVA, History, and Continuation History heuristics.
*   **Data Structures**: Shared Transposition Table, Syzygy Endgame Tablebases support.

### Evaluation
The engine implements two evaluation Contexts:
*   **Neural Evaluation**: An ONNX-integrated Mixture of Experts (MoE) network taking input from a custom bitboard feature extractor. 
*   **Heuristic Fallback**: A traditional static evaluation relying on PeSTO Piece-Square Tables with game-phase interpolation.

## Targets and Compilation

The project uses CMake for build configuration. The native target dynamically links against `onnxruntime`, while the WebAssembly target statically links the dependencies for execution in web workers.

### Requirements
*   **CMake**: `3.14+`
*   **Compiler**: `g++` or `clang` (C++20 required)

The engine requires `onnxruntime` (v1.23.0) for its neural network evaluation. You can automatically download the correct pre-compiled binaries for your operating system (macOS/Linux) by running the included setup script:

```bash
cd eval_model_onnx
./download_onnxruntime.sh
cd ../engine
```

### Native Build (Linux / macOS)

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

The compiled binary `chess_engine` will be available in the `build` directory.

### WebAssembly Build (Emscripten)

The WASM build uses Emscripten and relies on `SharedArrayBuffer` for threading support (Pthreads). 

```bash
cd engine/
mkdir build_wasm && cd build_wasm
emcmake cmake -DCMAKE_BUILD_TYPE=Release ..
emmake make -j$(nproc)
```

## Usage

**io-chess** communicates via the Universal Chess Interface (UCI) protocol and can be attached to standard GUIs (e.g. Cute Chess, Arena).

```bash
./chess_engine
```

**Supported UCI Commands:**
*   `uci`: Identify engine and display parameters.
*   `isready`: Initialize internal memory and verify readiness.
*   `position [startpos | fen <fen>] moves <moves>`: Configure the board state.
*   `go [depth <limit> | wtime <ms> btime <ms>]`: Begin search algorithm.
*   `quit`: Terminate the process.