#include "ExpertRouter.hpp"
#include "FeatureExtractor.hpp"
#include "WDLNormalizer.hpp"
#include "Writers.hpp"
#include "chess.hpp"
#include "progressbar.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  std::string fens_file =
      "/home/io/USI/io-bot/io-chess_engine/nnue/data/chessData.csv";
  std::string output_dir = "../data/processed";

  // Parse command line args
  if (argc > 1) {
    fens_file = argv[1];
  }
  if (argc > 2) {
    output_dir = argv[2];
  }

  // Create output directories
  std::string train_dir = output_dir + "/train";
  std::string val_dir = output_dir + "/val";

  fs::create_directories(train_dir);
  fs::create_directories(val_dir);

  // 1. Count lines for progress bar
  std::cout << "Counting lines in " << fens_file << "..." << std::endl;
  std::ifstream file_count(fens_file);
  if (!file_count.is_open()) {
    std::cerr << "Error: Could not open file " << fens_file << std::endl;
    return 1;
  }
  file_count.unsetf(std::ios_base::skipws);
  size_t total_lines = std::count(std::istreambuf_iterator<char>(file_count),
                                  std::istreambuf_iterator<char>(), '\n');
  if (total_lines > 0)
    total_lines--; // Subtract header
  std::cout << "Found " << total_lines << " examples." << std::endl;

  // 2. Setup Writers for train and val separately
  const size_t BATCH_CAPACITY = 10000;

  // Train writers (global)
  DatasetWriter train_feature_writer(train_dir + "/features.bin",
                                     BATCH_CAPACITY);
  WDLWriter train_wdl_writer(train_dir + "/labels.bin", BATCH_CAPACITY);
  ExpertWeightsWriter train_expert_weight_writer(
      train_dir + "/expert_weights.bin", BATCH_CAPACITY);

  // Val writers (global)
  DatasetWriter val_feature_writer(val_dir + "/features.bin", BATCH_CAPACITY);
  WDLWriter val_wdl_writer(val_dir + "/labels.bin", BATCH_CAPACITY);
  ExpertWeightsWriter val_expert_weight_writer(val_dir + "/expert_weights.bin",
                                               BATCH_CAPACITY);

  // Per-expert writers (train and val)
  ExpertDatasetWriter train_expert_datasets(train_dir, BATCH_CAPACITY);
  ExpertDatasetWriter val_expert_datasets(val_dir, BATCH_CAPACITY);

  // 3. Process File
  std::ifstream file(fens_file);
  std::string line;
  std::getline(file, line); // Skip header

  // Create reusable structs (optimization)
  ChessInput features;
  ExpertRouter::ExpertWeights expert_weights;

  ProgressBar pbar(total_lines, 50, false);
  size_t processed = 0;
  size_t train_count = 0;
  size_t val_count = 0;

  while (std::getline(file, line)) {
    size_t comma_pos = line.find(',');
    if (comma_pos == std::string::npos)
      continue;

    std::string fen = line.substr(0, comma_pos);
    std::string eval_str = line.substr(comma_pos + 1);

    // Parse Board & Features
    chess::Board board(fen);
    FeatureExtractor::fill_input(board, features);

    // Determine side to move
    bool is_white = board.sideToMove() == chess::Color::WHITE;

    // Convert eval to WDL + MateDistance (Side-to-Move perspective)
    WDLOutput wdl = WDLNormalizer::convert(eval_str, is_white);

    // Get raw centipawns for ExpertRouter (STM perspective)
    float eval_cp = WDLNormalizer::to_centipawns(eval_str, is_white);

    // Compute Expert Weights (using centipawns)
    ExpertRouter::compute_weights(features, eval_cp, expert_weights);

    // Deterministic train/val split using Zobrist hash
    // This is 100% stable and uses standard chess hashing
    bool is_val = (board.zobrist() % 100 < 10);

    if (is_val) {
      val_feature_writer.add(features);
      val_wdl_writer.add(wdl);
      val_expert_weight_writer.add(expert_weights);
      val_expert_datasets.add(features, wdl, expert_weights);
      val_count++;
    } else {
      train_feature_writer.add(features);
      train_wdl_writer.add(wdl);
      train_expert_weight_writer.add(expert_weights);
      train_expert_datasets.add(features, wdl, expert_weights);
      train_count++;
    }

    processed++;
    pbar.update(processed);
  }
  pbar.complete();

  // Print summary
  std::cout << "\n=== Dataset Summary ===" << std::endl;
  std::cout << "Total processed: " << processed << std::endl;
  std::cout << "Train samples:   " << train_count << " ("
            << 100.0 * train_count / processed << "%)" << std::endl;
  std::cout << "Val samples:     " << val_count << " ("
            << 100.0 * val_count / processed << "%)" << std::endl;

  std::cout << "\n--- Train Directory: " << train_dir << " ---" << std::endl;
  std::cout << "Global files:" << std::endl;
  std::cout << "  - features.bin       ("
            << train_count * sizeof(ChessInput) / (1024 * 1024) << " MB)"
            << std::endl;
  std::cout << "  - labels.bin         ("
            << train_count * 4 * sizeof(float) / (1024 * 1024) << " MB)"
            << std::endl;
  std::cout << "  - expert_weights.bin ("
            << train_count * 6 * sizeof(float) / (1024 * 1024) << " MB)"
            << std::endl;
  std::cout << "Per-expert files:" << std::endl;
  train_expert_datasets.print_stats();

  std::cout << "\n--- Val Directory: " << val_dir << " ---" << std::endl;
  std::cout << "Global files:" << std::endl;
  std::cout << "  - features.bin       ("
            << val_count * sizeof(ChessInput) / (1024 * 1024) << " MB)"
            << std::endl;
  std::cout << "  - labels.bin         ("
            << val_count * 4 * sizeof(float) / (1024 * 1024) << " MB)"
            << std::endl;
  std::cout << "  - expert_weights.bin ("
            << val_count * 6 * sizeof(float) / (1024 * 1024) << " MB)"
            << std::endl;
  std::cout << "Per-expert files:" << std::endl;
  val_expert_datasets.print_stats();

  return 0;
}
