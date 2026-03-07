// benchmark_eval.cpp - Benchmark neural network evaluation performance
// Tests board-to-features conversion and model inference speed

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <random>
#include <thread>
#include <atomic>
#include "Types.h"
#include "eval/Evaluator.h"
#include "eval/EvalContextMoE.h"
#include "FeatureExtractor.hpp"

using namespace std::chrono;

struct BenchmarkResult {
    size_t numPositions;
    double totalTime;
    double avgTime;
    double featureTime;
    double inferenceTime;
    double positionsPerSecond;
};

struct PlayoutResult {
    size_t plies;
    double totalTime;
    double avgTime;
    double throughput;
};

struct PlayoutBatchResult {
    size_t games;
    size_t totalPlies;
    double totalTime;
    double avgPliesPerGame;
    double avgTimePerGame;
    double avgTimePerPly;
    double pliesPerSecond;
};

// Load FEN positions from CSV file
std::vector<std::string> loadFens(const std::string& csvPath, size_t maxPositions = 0) {
    std::vector<std::string> fens;
    std::ifstream file(csvPath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << csvPath << std::endl;
        return fens;
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        if (maxPositions > 0 && fens.size() >= maxPositions) {
            break;
        }
        
        // Parse CSV: FEN,Evaluation
        size_t commaPos = line.find(',');
        if (commaPos != std::string::npos) {
            std::string fen = line.substr(0, commaPos);
            fens.push_back(fen);
        }
    }
    
    file.close();
    return fens;
}

// Benchmark feature extraction only
BenchmarkResult benchmarkFeatureExtraction(const std::vector<std::string>& fens) {
    BenchmarkResult result{};
    result.numPositions = fens.size();
    
    ChessInput features{};
    std::vector<uint8_t> dummyFeatures(32 * 64);
    
    auto start = high_resolution_clock::now();
    
    for (const auto& fen : fens) {
        Board board;
        board.setFen(fen);
        FeatureExtractor::fill_input(board, features);
        // Access data to prevent optimization
        dummyFeatures[0] = features.layers[0][0];
    }
    
    auto end = high_resolution_clock::now();
    
    result.totalTime = duration_cast<microseconds>(end - start).count() / 1000.0; // ms
    result.avgTime = result.totalTime / result.numPositions;
    result.positionsPerSecond = (result.numPositions * 1000.0) / result.totalTime;
    
    return result;
}

// Benchmark a single random playout starting from a given FEN.
// For each ply: evaluate position, generate legal moves, pick one at random, make it, repeat.
PlayoutResult benchmarkRandomPlayout(const std::string& startFen, size_t plies, IEvaluator& evaluator, std::mt19937& rng) {
    PlayoutResult result{};
    result.plies = 0;
    
    Board board;
    board.setFen(startFen);
    
    auto start = high_resolution_clock::now();
    
    for (size_t i = 0; i < plies; ++i) {
        // Evaluate current position
        auto evalStart = high_resolution_clock::now();
        float eval = evaluator.evaluate(board);
        auto evalEnd = high_resolution_clock::now();
        (void)eval; // prevent unused warning
        
        // Generate legal moves
        Movelist moves;
        MoveGen::generateLegalMoves(moves, board);
        if (moves.size() == 0) {
            break; // game over
        }
        
        // Pick a random legal move
        std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
        Move move = moves[dist(rng)];
        
        // Make the move and continue
        board.makeMove(move);
        result.plies++;
    }
    
    auto end = high_resolution_clock::now();
    result.totalTime = duration_cast<microseconds>(end - start).count() / 1000.0; // ms
    if (result.plies > 0) {
        result.avgTime = result.totalTime / result.plies;
        result.throughput = (result.plies * 1000.0) / result.totalTime;
    }
    
    return result;
}

// Benchmark multiple random playouts sampled from a set of FENs
PlayoutBatchResult benchmarkRandomPlayouts(const std::vector<std::string>& fens, size_t games, size_t plies, IEvaluator& evaluator) {
    PlayoutBatchResult batch{};
    batch.games = std::min(games, fens.size());
    if (batch.games == 0) return batch;
    
    std::mt19937 rng(42); // deterministic for reproducibility
    std::uniform_int_distribution<size_t> fenDist(0, fens.size() - 1);
    
    auto start = high_resolution_clock::now();
    
    for (size_t i = 0; i < batch.games; ++i) {
        // Pick a random starting FEN from the list
        const std::string& fen = fens[fenDist(rng)];
        auto res = benchmarkRandomPlayout(fen, plies, evaluator, rng);
        batch.totalPlies += res.plies;
    }
    
    auto end = high_resolution_clock::now();
    batch.totalTime = duration_cast<microseconds>(end - start).count() / 1000.0; // ms
    batch.avgPliesPerGame = static_cast<double>(batch.totalPlies) / batch.games;
    batch.avgTimePerGame = batch.totalTime / batch.games;
    if (batch.totalPlies > 0) {
        batch.avgTimePerPly = batch.totalTime / batch.totalPlies;
        batch.pliesPerSecond = (batch.totalPlies * 1000.0) / batch.totalTime;
    }
    return batch;
}

// Benchmark full evaluation (feature extraction + inference)
BenchmarkResult benchmarkFullEvaluation(
    const std::vector<std::string>& fens, 
    IEvaluator& evaluator,
    bool separateTiming = true
) {
    BenchmarkResult result{};
    result.numPositions = fens.size();
    
    double totalFeatureTime = 0.0;
    double totalInferenceTime = 0.0;
    
    auto overallStart = high_resolution_clock::now();
    
    for (const auto& fen : fens) {
        Board board;
        board.setFen(fen);
        
        if (separateTiming) {
            // Measure feature extraction
            ChessInput features{};
            auto featStart = high_resolution_clock::now();
            FeatureExtractor::fill_input(board, features);
            auto featEnd = high_resolution_clock::now();
            totalFeatureTime += duration_cast<nanoseconds>(featEnd - featStart).count() / 1000.0; // microseconds
            
            // Measure inference (by calling evaluate which includes feature extraction again)
            // This is not perfect but gives us an idea
            auto infStart = high_resolution_clock::now();
            float eval = evaluator.evaluate(board);
            auto infEnd = high_resolution_clock::now();
            double totalCallTime = duration_cast<nanoseconds>(infEnd - infStart).count() / 1000.0; // microseconds
            totalInferenceTime += (totalCallTime - duration_cast<nanoseconds>(featEnd - featStart).count() / 1000.0);
            
            // Prevent optimization
            if (eval > 1e10) std::cout << eval;
        } else {
            // Just measure total evaluation time
            float eval = evaluator.evaluate(board);
            // Prevent optimization
            if (eval > 1e10) std::cout << eval;
        }
    }
    
    auto overallEnd = high_resolution_clock::now();
    
    result.totalTime = duration_cast<microseconds>(overallEnd - overallStart).count() / 1000.0; // ms
    result.avgTime = result.totalTime / result.numPositions;
    result.featureTime = totalFeatureTime / 1000.0; // convert to ms
    result.inferenceTime = totalInferenceTime / 1000.0; // convert to ms
    result.positionsPerSecond = (result.numPositions * 1000.0) / result.totalTime;
    
    return result;
}

// Benchmark multi-threaded evaluation (Lazy SMP simulation)
BenchmarkResult benchmarkMultiThreaded(
    const std::vector<std::string>& fens, 
    Evaluator& evaluator,
    size_t numThreads
) {
    BenchmarkResult result{};
    result.numPositions = fens.size();
    
    // Split work among threads
    std::vector<std::thread> threads;
    std::atomic<size_t> positionsProcessed{0};
    
    auto overallStart = high_resolution_clock::now();
    
    size_t fensPerThread = (fens.size() + numThreads - 1) / numThreads;
    
    for (size_t t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            // Create thread-local context (Lazy SMP)
            auto ctx = evaluator.createThreadContext();
            
            size_t startIdx = t * fensPerThread;
            size_t endIdx = std::min(startIdx + fensPerThread, fens.size());
            
            if (startIdx >= fens.size()) return;
            
            for (size_t i = startIdx; i < endIdx; ++i) {
                Board board;
                board.setFen(fens[i]);
                float eval = ctx->evaluate(board);
                if (eval > 1e10) std::cout << eval; // Prevent optimization
            }
            
            positionsProcessed += (endIdx - startIdx);
        });
    }
    
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
    
    auto overallEnd = high_resolution_clock::now();
    
    result.totalTime = duration_cast<microseconds>(overallEnd - overallStart).count() / 1000.0; // ms
    result.avgTime = result.totalTime / result.numPositions;
    result.positionsPerSecond = (result.numPositions * 1000.0) / result.totalTime;
    
    return result;
}

void printResults(const std::string& name, const BenchmarkResult& result) {
    std::cout << "\n=== " << name << " ===" << std::endl;
    std::cout << "Positions:     " << result.numPositions << std::endl;
    std::cout << "Total time:    " << std::fixed << std::setprecision(2) << result.totalTime << " ms" << std::endl;
    std::cout << "Avg per pos:   " << std::fixed << std::setprecision(4) << result.avgTime << " ms" << std::endl;
    
    if (result.featureTime > 0) {
        double avgFeatureTime = result.featureTime / result.numPositions;
        std::cout << "Feature time:  " << std::fixed << std::setprecision(2) << result.featureTime << " ms"
                  << " (" << std::fixed << std::setprecision(4) << avgFeatureTime << " ms/pos)" << std::endl;
    }
    
    if (result.inferenceTime > 0) {
        double avgInferenceTime = result.inferenceTime / result.numPositions;
        std::cout << "Inference time:" << std::fixed << std::setprecision(2) << result.inferenceTime << " ms"
                  << " (" << std::fixed << std::setprecision(4) << avgInferenceTime << " ms/pos)" << std::endl;
    }
    
    std::cout << "Throughput:    " << std::fixed << std::setprecision(1) << result.positionsPerSecond << " pos/s" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "===========================================================" << std::endl;
    std::cout << "   Neural Network Evaluator Benchmark" << std::endl;
    std::cout << "===========================================================" << std::endl;
    
    // Parse arguments
    std::string modelPath;
    std::string csvPath = "../data/random_evals.csv";
    size_t numPositions = 1000;
    size_t numThreads = 1;
    size_t playoutCount = 0; // if >0, run playout benchmark using sampled FENs
    size_t playoutPlies = 50;
    bool moeBreakdown = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--csv" && i + 1 < argc) {
            csvPath = argv[++i];
        } else if (arg == "--num" && i + 1 < argc) {
            numPositions = std::stoul(argv[++i]);
        } else if (arg == "--playout-count" && i + 1 < argc) {
            playoutCount = std::stoul(argv[++i]);
        } else if (arg == "--playout-moves" && i + 1 < argc) {
            playoutPlies = std::stoul(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            numThreads = std::stoul(argv[++i]);
        } else if (arg == "--moe-breakdown") {
            moeBreakdown = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "\nUsage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --model <path>   Path to ONNX model (required)\n"
                      << "  --csv <path>     Path to CSV file with FEN positions\n"
                      << "                   (default: ../data/random_evals.csv)\n"
                      << "  --num <n>        Number of positions to evaluate (default: 1000)\n"
                      << "  --threads <n>    Number of threads for evaluation (default: 1)\n"
                      << "  --playout-count <n>   Run N random playouts sampled from the loaded FEN list\n"
                      << "  --playout-moves <n>   Number of plies per playout (default: 50)\n"
                      << "  --moe-breakdown       Profile backbone vs expert latency\n"
                      << "  --help, -h       Show this help message\n"
                      << std::endl;
            return 0;
        }
    }
    
    if (modelPath.empty()) {
        std::cerr << "\nError: Model path is required!\n"
                  << "Usage: " << argv[0] << " --model <path> [--csv <path>] [--num <n>]\n"
                  << "Run with --help for more information.\n" << std::endl;
        return 1;
    }
    
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Model:      " << modelPath << std::endl;
    std::cout << "  CSV:        " << csvPath << std::endl;
    std::cout << "  Positions:  " << numPositions << std::endl;
    std::cout << "  Threads:    " << numThreads << std::endl;
    if (playoutCount > 0) {
        std::cout << "  Playouts:   " << playoutCount << " games, " << playoutPlies << " plies each" << std::endl;
    }
    
    // Load FEN positions
    std::cout << "\nLoading FEN positions from CSV..." << std::endl;
    auto fens = loadFens(csvPath, numPositions);
    
    if (fens.empty()) {
        std::cerr << "Error: No FEN positions loaded!" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << fens.size() << " positions" << std::endl;
    
    // Load model
    std::cout << "\nLoading ONNX model..." << std::endl;
    try {
        Evaluator evaluator(modelPath, numThreads);
        
        if (!evaluator.isLoaded()) {
            std::cerr << "Error: Failed to load model!" << std::endl;
            return 1;
        }
        
        std::cout << "Model loaded successfully" << std::endl;
        
        // Create evaluation context
        auto evalContext = evaluator.createThreadContext();
        
        // Warmup (run a few evaluations to stabilize timing)
        std::cout << "\nWarming up (10 positions)..." << std::endl;
        size_t warmupSize = std::min(size_t(10), fens.size());
        for (size_t i = 0; i < warmupSize; ++i) {
            Board board;
            board.setFen(fens[i]);
            evalContext->evaluate(board);
        }
        std::cout << "Warmup complete" << std::endl;
        
        // Test: Print 10 sample evaluations
        std::cout << "\n=== Sample Evaluations (10 positions) ===" << std::endl;
        // Benchmark 1: Feature extraction only
        std::cout << "\n[1/2] Benchmarking feature extraction..." << std::endl;
        auto featureResult = benchmarkFeatureExtraction(fens);
        printResults("Feature Extraction Only", featureResult);
        
        // Benchmark 2: Full evaluation
        std::cout << "\n[2/2] Benchmarking full evaluation (" << numThreads << " threads)..." << std::endl;
        BenchmarkResult fullResult;
        
        if (numThreads > 1) {
             fullResult = benchmarkMultiThreaded(fens, evaluator, numThreads);
             printResults("Multi-threaded Evaluation", fullResult);
        } else {
             fullResult = benchmarkFullEvaluation(fens, *evalContext, false);
             printResults("Full Evaluation (Feature + Inference)", fullResult);
        }
        
        // Calculate inference-only estimate
        std::cout << "\n=== Inference-Only Estimate ===" << std::endl;
        double inferenceOnlyTime = fullResult.totalTime - featureResult.totalTime;
        double avgInferenceTime = inferenceOnlyTime / fullResult.numPositions;
        double inferencePositionsPerSecond = (fullResult.numPositions * 1000.0) / inferenceOnlyTime;
        
        std::cout << "Total time:    " << std::fixed << std::setprecision(2) << inferenceOnlyTime << " ms" << std::endl;
        std::cout << "Avg per pos:   " << std::fixed << std::setprecision(4) << avgInferenceTime << " ms" << std::endl;
        std::cout << "Throughput:    " << std::fixed << std::setprecision(1) << inferencePositionsPerSecond << " pos/s" << std::endl;
        
        // Summary
        std::cout << "\n===========================================================" << std::endl;
        std::cout << "   Summary" << std::endl;
        std::cout << "===========================================================" << std::endl;
        std::cout << "Feature extraction:    " << std::fixed << std::setprecision(4) 
                  << (featureResult.avgTime) << " ms/pos ("
                  << std::fixed << std::setprecision(1) << (featureResult.avgTime / fullResult.avgTime * 100) << "% of total)" << std::endl;
        std::cout << "Model inference:       " << std::fixed << std::setprecision(4) 
                  << avgInferenceTime << " ms/pos ("
                  << std::fixed << std::setprecision(1) << (avgInferenceTime / fullResult.avgTime * 100) << "% of total)" << std::endl;
        std::cout << "Total evaluation:      " << std::fixed << std::setprecision(4) 
                  << fullResult.avgTime << " ms/pos" << std::endl;
        std::cout << "Overall throughput:    " << std::fixed << std::setprecision(1) 
                  << fullResult.positionsPerSecond << " pos/s" << std::endl;
        std::cout << "===========================================================" << std::endl;

        // Optional: random playout benchmark (game-like scenario) using sampled FENs
        if (playoutCount > 0) {
            std::cout << "\n[3/3] Benchmarking random playouts (sampled from loaded FENs)..." << std::endl;
            auto playoutResult = benchmarkRandomPlayouts(fens, playoutCount, playoutPlies, *evalContext);
            std::cout << "\n=== Random Playouts ===" << std::endl;
            std::cout << "Games played:   " << playoutResult.games << std::endl;
            std::cout << "Total plies:    " << playoutResult.totalPlies << std::endl;
            std::cout << "Total time:     " << std::fixed << std::setprecision(2) << playoutResult.totalTime << " ms" << std::endl;
            std::cout << "Avg plies/game: " << std::fixed << std::setprecision(2) << playoutResult.avgPliesPerGame << std::endl;
            std::cout << "Avg time/game:  " << std::fixed << std::setprecision(2) << playoutResult.avgTimePerGame << " ms" << std::endl;
            std::cout << "Avg time/ply:   " << std::fixed << std::setprecision(4) << playoutResult.avgTimePerPly << " ms" << std::endl;
            std::cout << "Throughput:     " << std::fixed << std::setprecision(1) << playoutResult.pliesPerSecond << " plies/s" << std::endl;
        }

        // MoE Component Breakdown Benchmark
        if (moeBreakdown) {
            std::cout << "\n[MoE] Benchmarking backbone vs expert breakdown..." << std::endl;

            // Get MoE context (downcast from IEvaluator)
            auto moeCtx = dynamic_cast<EvalContextMoE*>(evalContext.get());
            if (!moeCtx) {
                std::cerr << "Error: Model is not MoE type, cannot run --moe-breakdown" << std::endl;
            } else {
                // Accumulate timings
                double sumFeat = 0, sumBB = 0, sumE[4] = {0}, sumWdl = 0, sumTotal = 0;
                int expertUsage[4] = {0}; // Count how often each expert is top-1
                size_t n = fens.size();

                for (size_t i = 0; i < n; ++i) {
                    Board board;
                    board.setFen(fens[i]);
                    auto t = moeCtx->benchmarkComponents(board);
                    sumFeat += t.featureExtractUs;
                    sumBB += t.backboneUs;
                    sumE[0] += t.expert0Us;
                    sumE[1] += t.expert1Us;
                    sumE[2] += t.expert2Us;
                    sumE[3] += t.expert3Us;
                    sumWdl += t.wdlConvertUs;
                    sumTotal += t.totalUs;
                    if (t.topExpert0 >= 0 && t.topExpert0 < 4) expertUsage[t.topExpert0]++;
                }

                const char* expertNames[4] = {"Tactical", "Strategic", "MajorEnd", "MinorEnd"};

                std::cout << "\n=== MoE Component Breakdown (" << n << " positions) ===" << std::endl;
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "  Feature Extract:  " << std::setw(8) << sumFeat / n << " μs/pos  ("
                          << std::setw(5) << std::setprecision(1) << (sumFeat / sumTotal * 100) << "%)" << std::endl;
                std::cout << "  Backbone:         " << std::setw(8) << std::setprecision(2) << sumBB / n << " μs/pos  ("
                          << std::setw(5) << std::setprecision(1) << (sumBB / sumTotal * 100) << "%)" << std::endl;
                for (int e = 0; e < 4; ++e) {
                    std::cout << "  Expert " << e << " (" << std::setw(9) << expertNames[e] << "): "
                              << std::setw(8) << std::setprecision(2) << sumE[e] / n << " μs/pos  ("
                              << std::setw(5) << std::setprecision(1) << (sumE[e] / sumTotal * 100) << "%)"
                              << "  top-1: " << std::setw(5) << std::setprecision(1) << (100.0 * expertUsage[e] / n) << "%" << std::endl;
                }
                std::cout << "  WDL Convert:      " << std::setw(8) << std::setprecision(2) << sumWdl / n << " μs/pos  ("
                          << std::setw(5) << std::setprecision(1) << (sumWdl / sumTotal * 100) << "%)" << std::endl;
                std::cout << "  ─────────────────────────────────────────" << std::endl;
                std::cout << "  Total:            " << std::setw(8) << std::setprecision(2) << sumTotal / n << " μs/pos" << std::endl;
                std::cout << "  All 4 experts:    " << std::setw(8) << std::setprecision(2) << (sumE[0]+sumE[1]+sumE[2]+sumE[3]) / n << " μs/pos  ("
                          << std::setw(5) << std::setprecision(1) << ((sumE[0]+sumE[1]+sumE[2]+sumE[3]) / sumTotal * 100) << "%)" << std::endl;
                std::cout << "  Backbone ratio:   " << std::setw(5) << std::setprecision(1) 
                          << (sumBB / (sumBB + sumE[0]+sumE[1]+sumE[2]+sumE[3]) * 100) << "% of inference" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
