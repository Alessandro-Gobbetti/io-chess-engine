// src/tools/compare_evals.cpp
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../Types.h"
#include "eval/Evaluator.h"
#include "eval/SimpleEvalContext.h"
#include "eval/WDLConverter.hpp"

// Simple command line parser
struct Config {
  std::string modelPath;
  std::string outputPath = "comparison.csv";
  int numPositions = 100000;
};

void printUsage(const char *name) {
  std::cout << "Usage: " << name
            << " --model <path> [--out <file>] [--num <N>]\n";
}

int main(int argc, char *argv[]) {
  Config config;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      config.modelPath = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      config.outputPath = argv[++i];
    } else if (arg == "--num" && i + 1 < argc) {
      config.numPositions = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      printUsage(argv[0]);
      return 0;
    }
  }

  if (config.modelPath.empty()) {
    std::cerr << "Error: --model required\n";
    printUsage(argv[0]);
    return 1;
  }

  std::cout << "Loading model from " << config.modelPath << "..." << std::endl;

  try {
    // Initialize evaluators
    // Initialize evaluators
    Evaluator nnEvaluatorManager(config.modelPath, 1, false);
    auto nnEval = nnEvaluatorManager.createThreadContext();

    SimpleEvalContext simpleEval;

    std::cout << "Starting comparison on " << config.numPositions
              << " positions..." << std::endl;

    std::ofstream out(config.outputPath);
    out << "fen,SimpleCP,NNCP,Phase" << std::endl;

    Board board; // Start position

    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(0, 255); // For move selection

    Movelist moves;

    int currentPlies = 0;

    // Phase constants (mirrored from SimpleEvalContext)
    const int PHASE_KNIGHT = 1;
    const int PHASE_BISHOP = 1;
    const int PHASE_ROOK = 2;
    const int PHASE_QUEEN = 4;

    auto getPhase = [&](const Board &b) {
      int phase = 0;
      phase += b.pieces(PieceType::KNIGHT).count() * PHASE_KNIGHT;
      phase += b.pieces(PieceType::BISHOP).count() * PHASE_BISHOP;
      phase += b.pieces(PieceType::ROOK).count() * PHASE_ROOK;
      phase += b.pieces(PieceType::QUEEN).count() * PHASE_QUEEN;
      return std::min(phase, 24); // Cap at 24
    };

    int positionsProcessed = 0;
    int gamesReplayed = 0;

    while (positionsProcessed < config.numPositions) {
      // Check if game over or too long
      MoveGen::generateLegalMoves(moves, board);

      if (moves.empty() || board.halfMoveClock() >= 100 || currentPlies > 200) {
        board = Board(); // Reset
        currentPlies = 0;
        gamesReplayed++;
        continue;
      }

      // Select random move
      std::uniform_int_distribution<int> moveIdx(0, moves.size() - 1);
      board.makeMove(moves[moveIdx(rng)]);
      currentPlies++;

      // Evaluate
      float simpleScore = simpleEval.evaluate(board);
      float nnScore = nnEval->evaluate(board);
      int phase = getPhase(board);

      // WDL conversion happens inside NN eval? No, IEvaluator returns float.
      // We assume it returns CP or comparable.
      // If NN returns WDL, we might need to convert.
      // But EvalContext usually returns CP if configured or pure output.
      // Let's assume CP for now as per previous checks.

      out << board.getFen() << "," << simpleScore << "," << nnScore << ","
          << phase << "\n";

      positionsProcessed++;
      if (positionsProcessed % 1000 == 0) {
        std::cout << "\rProcessed " << positionsProcessed << "/"
                  << config.numPositions << " ("
                  << (positionsProcessed * 100 / config.numPositions) << "%)"
                  << std::flush;
      }
    }

    std::cout << "\nDone! Saved to " << config.outputPath << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "\nError: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
