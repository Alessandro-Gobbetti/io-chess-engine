/**
 * @file main.cpp
 * @brief Main entry point for the preprocessing pipeline.
 */
#include "ExpertRouter.hpp"
#include "FactorizedFeatureExtractor.hpp"
#include "FeatureExtractor.hpp"
#include "WDLNormalizer.hpp"
#include "Writers.hpp"
#include "chess.hpp"
#include "progressbar.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

enum class FeatureMode { Standard, Factorized };

constexpr std::array<uint8_t, 12> kPackedPlanesPerGroup = [] {
  std::array<uint8_t, 12> out{};
  for (int g = 0; g < 12; ++g) {
    out[g] = static_cast<uint8_t>(FactorizedFeatureExtractor::PLANES_PER_TYPE[g % 6]);
  }
  return out;
}();

constexpr std::array<uint16_t, 12> kPackedPlaneOffsets = [] {
  std::array<uint16_t, 12> out{};
  uint16_t offset = 0;
  for (int g = 0; g < 12; ++g) {
    out[g] = offset;
    offset += kPackedPlanesPerGroup[g];
  }
  return out;
}();

constexpr int kPackedBranchPlaneTotal = [] {
  int total = 0;
  for (int g = 0; g < 12; ++g) {
    total += static_cast<int>(kPackedPlanesPerGroup[g]);
  }
  return total;
}();
static_assert(kPackedPlaneOffsets[11] + kPackedPlanesPerGroup[11] ==
                  kPackedBranchPlaneTotal,
              "Packed branch layout mismatch");

struct alignas(64) PackedFactorizedInput {
  uint64_t branches[kPackedBranchPlaneTotal];
  uint8_t bypass_continuous[2][64]; // US_KING_DIST(0), THEM_KING_DIST(1)
  uint64_t bypass_categorical[10];  // Remaining boolean bypass layers
  float global[32];
};

constexpr float kAuxGateThreshold = 0.1f;
constexpr float kBaseExpertThreshold = 1.8f;

void print_usage(const char *exe_name) {
  std::cout << "Usage: " << exe_name
            << " [fens_file] [output_dir] [--factorized|--standard] [--limit N] [--skip N]"
            << std::endl;
  std::cout << "  Default extractor: factorized (PackedFactorizedInput)" << std::endl;
  std::cout << "  Default input: ../data/shuffled.csv" << std::endl;
  std::cout << "  Optional flags:" << std::endl;
  std::cout << "    --factorized, -f : write factorized features (default)"
            << std::endl;
  std::cout << "    --standard        : write standard features" << std::endl;
  std::cout << "    --limit, -l N     : only process the first N samples" << std::endl;
  std::cout << "    --skip, -s N      : skip the first N input rows (after header)"
            << std::endl;
  std::cout << "    --factorized-global-only : in factorized mode, skip"
               " per-expert factorized feature files"
            << std::endl;
  std::cout << "    --factorized-use-standard-router : in factorized mode,"
               " route experts using standard features"
            << std::endl;
}

void pack_factorized_input(const FactorizedInput &in, PackedFactorizedInput &out) {
  auto q01 = [](float v) -> uint8_t {
    const float clamped = std::clamp(v, 0.0f, 1.0f);
    return static_cast<uint8_t>(clamped * 255.0f + 0.5f);
  };
  
  auto pack_bits = [](const float* plane) -> uint64_t {
    uint64_t bb = 0;
    for (int sq = 0; sq < 64; ++sq) {
      if (plane[sq] > 0.5f) {
        bb |= (1ULL << sq);
      }
    }
    return bb;
  };

  for (int g = 0; g < 12; ++g) {
    const int offset = kPackedPlaneOffsets[g];
    const int planes = kPackedPlanesPerGroup[g];
    for (int p = 0; p < planes; ++p) {
      out.branches[offset + p] = pack_bits(in.branches[g][p]);
    }
    
    if (g < 2) {
      for (int sq = 0; sq < 64; ++sq) {
        out.bypass_continuous[g][sq] = q01(in.bypass[g][sq]);
      }
    } else {
      out.bypass_categorical[g - 2] = pack_bits(in.bypass[g]);
    }
  }
  std::memcpy(out.global, in.global, sizeof(out.global));
}

void flush_factorized_buffer(const std::string &filename,
                             std::vector<PackedFactorizedInput> &buffer) {
  if (buffer.empty()) {
    return;
  }

  std::ofstream file(filename, std::ios::binary | std::ios::app);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file for append " << filename
              << std::endl;
    std::exit(1);
  }

  file.write(reinterpret_cast<const char *>(buffer.data()),
             buffer.size() * sizeof(PackedFactorizedInput));
  buffer.clear();
}

void flush_index_buffer(const std::string &filename,
                        std::vector<uint32_t> &buffer) {
  if (buffer.empty()) {
    return;
  }

  std::ofstream file(filename, std::ios::binary | std::ios::app);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file for append " << filename
              << std::endl;
    std::exit(1);
  }

  file.write(reinterpret_cast<const char *>(buffer.data()),
             buffer.size() * sizeof(uint32_t));
  buffer.clear();
}

} // namespace

int main(int argc, char **argv) {
  std::string fens_file =
      "../data/shuffled.csv";
  std::string output_dir = "../data/processed";
  FeatureMode feature_mode = FeatureMode::Factorized;
  bool factorized_global_only = false;
  bool factorized_use_standard_router = false;
  size_t limit = 0;
  size_t skip = 0;

  // Parse command line args
  std::vector<std::string> positional_args;
  positional_args.reserve(2);

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--factorized" || arg == "-f") {
      feature_mode = FeatureMode::Factorized;
      continue;
    }
    if (arg == "--standard") {
      feature_mode = FeatureMode::Standard;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
    if (arg == "--factorized-global-only") {
      factorized_global_only = true;
      continue;
    }
    if (arg == "--factorized-use-standard-router") {
      factorized_use_standard_router = true;
      continue;
    }

    if (arg == "--limit" || arg == "-l") {
      if (i + 1 < argc) {
        limit = std::stoull(argv[++i]);
      } else {
        std::cerr << "Error: --limit requires a value" << std::endl;
        return 1;
      }
      continue;
    }

    if (arg == "--skip" || arg == "-s") {
      if (i + 1 < argc) {
        skip = std::stoull(argv[++i]);
      } else {
        std::cerr << "Error: --skip requires a value" << std::endl;
        return 1;
      }
      continue;
    }

    positional_args.push_back(arg);
  }

  if (!positional_args.empty()) {
    fens_file = positional_args[0];
  }
  if (positional_args.size() >= 2) {
    output_dir = positional_args[1];
  }
  if (positional_args.size() > 2) {
    std::cerr << "Error: Too many positional arguments." << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  // Create output directories
  std::string train_dir = output_dir + "/train";
  std::string val_dir = output_dir + "/val";

  fs::create_directories(train_dir);
  fs::create_directories(val_dir);

  std::cout << "Feature mode: "
            << (feature_mode == FeatureMode::Standard ? "standard"
                                                      : "factorized")
            << std::endl;
  if (feature_mode == FeatureMode::Factorized && factorized_global_only) {
    std::cout << "Factorized output mode: global-only" << std::endl;
  }
  if (feature_mode == FeatureMode::Factorized) {
    std::cout << "Routing source: "
              << (factorized_use_standard_router ? "standard features"
                                                : "factorized features")
              << std::endl;
  }

  // 1. Count lines for progress bar
  size_t total_lines = 0;
  if (limit > 0) {
    total_lines = limit;
    if (skip > 0) {
      std::cout << "Skipping first " << skip
                << " rows, then processing up to " << limit
                << " samples from " << fens_file << "..." << std::endl;
    } else {
      std::cout << "Processing first " << limit << " samples from "
                << fens_file << "..." << std::endl;
    }
  } else {
    std::cout << "Counting lines in " << fens_file << "..." << std::endl;
    std::ifstream file_count(fens_file);
    if (!file_count.is_open()) {
      std::cerr << "Error: Could not open file " << fens_file << std::endl;
      return 1;
    }
    file_count.unsetf(std::ios_base::skipws);
    total_lines = std::count(std::istreambuf_iterator<char>(file_count),
                                    std::istreambuf_iterator<char>(), '\n');
    if (total_lines > 0)
      total_lines--; // Subtract header
    if (skip > 0) {
      if (skip >= total_lines) {
        total_lines = 0;
      } else {
        total_lines -= skip;
      }
      std::cout << "Skipping first " << skip << " rows after header." << std::endl;
    }
    std::cout << "Found " << total_lines << " examples." << std::endl;
  }

  // 2. Setup Writers for train and val separately
  const size_t BATCH_CAPACITY = 10000;

  // Train writers (global)
    std::unique_ptr<DatasetWriter> train_feature_writer;
  WDLWriter train_wdl_writer(train_dir + "/labels.bin", BATCH_CAPACITY);
  ExpertWeightsWriter train_expert_weight_writer(
      train_dir + "/expert_weights.bin", BATCH_CAPACITY);

  // Val writers (global)
    std::unique_ptr<DatasetWriter> val_feature_writer;
  WDLWriter val_wdl_writer(val_dir + "/labels.bin", BATCH_CAPACITY);
  ExpertWeightsWriter val_expert_weight_writer(val_dir + "/expert_weights.bin",
                                               BATCH_CAPACITY);

  // Per-expert writers (train and val)
    std::unique_ptr<ExpertDatasetWriter> train_expert_datasets;
    std::unique_ptr<ExpertDatasetWriter> val_expert_datasets;

    std::vector<PackedFactorizedInput> train_factorized_buffer;
    std::vector<PackedFactorizedInput> val_factorized_buffer;
    std::array<std::vector<uint32_t>, ExpertRouter::NUM_EXPERTS>
      train_factorized_expert_index_buffers;
    std::array<std::vector<uint32_t>, ExpertRouter::NUM_EXPERTS>
      val_factorized_expert_index_buffers;
    std::array<size_t, ExpertRouter::NUM_EXPERTS> train_expert_counts{};
    std::array<size_t, ExpertRouter::NUM_EXPERTS> val_expert_counts{};
    std::array<std::string, ExpertRouter::NUM_EXPERTS> train_expert_index_paths;
    std::array<std::string, ExpertRouter::NUM_EXPERTS> val_expert_index_paths;

    if (feature_mode == FeatureMode::Standard) {
    train_feature_writer =
      std::make_unique<DatasetWriter>(train_dir + "/features.bin",
                      BATCH_CAPACITY);
    val_feature_writer = std::make_unique<DatasetWriter>(
      val_dir + "/features.bin", BATCH_CAPACITY);

    train_expert_datasets =
      std::make_unique<ExpertDatasetWriter>(train_dir, BATCH_CAPACITY);
    val_expert_datasets =
      std::make_unique<ExpertDatasetWriter>(val_dir, BATCH_CAPACITY);
    } else {
    std::ofstream train_file(train_dir + "/features.bin", std::ios::binary);
    std::ofstream val_file(val_dir + "/features.bin", std::ios::binary);
    if (!train_file.is_open() || !val_file.is_open()) {
      std::cerr << "Error: Failed to create factorized feature files."
          << std::endl;
      return 1;
    }

    train_factorized_buffer.reserve(BATCH_CAPACITY);
    val_factorized_buffer.reserve(BATCH_CAPACITY);

    if (!factorized_global_only) {
      for (int i = 0; i < ExpertRouter::NUM_EXPERTS; ++i) {
        train_factorized_expert_index_buffers[i].reserve(BATCH_CAPACITY);
        val_factorized_expert_index_buffers[i].reserve(BATCH_CAPACITY);

        train_expert_index_paths[i] =
          train_dir + "/expert" + std::to_string(i) + "_indices.bin";
        val_expert_index_paths[i] =
          val_dir + "/expert" + std::to_string(i) + "_indices.bin";

        std::ofstream train_expert_file(train_expert_index_paths[i],
                                        std::ios::binary);
        std::ofstream val_expert_file(val_expert_index_paths[i],
                                      std::ios::binary);
        if (!train_expert_file.is_open() || !val_expert_file.is_open()) {
          std::cerr << "Error: Failed to create per-expert index files."
                    << std::endl;
          return 1;
        }
      }
    }
    }

  // 3. Process File
  std::ifstream file(fens_file);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << fens_file << std::endl;
    return 1;
  }
  std::string line;
  std::getline(file, line); // Skip header

  // Create reusable structs (optimization)
  ChessInput standard_features{};
  FactorizedInput factorized_features{};
  PackedFactorizedInput packed_factorized_features{};
  ExpertRouter::ExpertWeights expert_weights{};

  ProgressBar pbar(total_lines, 50, false);
  size_t processed = 0;
  size_t lines_read = 0;
  size_t train_count = 0;
  size_t val_count = 0;

  while (std::getline(file, line)) {
    lines_read++;
    if (lines_read <= skip) {
      continue;
    }
    if (limit > 0 && processed >= limit) {
      break;
    }
    size_t comma_pos = line.find(',');
    if (comma_pos == std::string::npos)
      continue;

    std::string fen = line.substr(0, comma_pos);
    std::string eval_str = line.substr(comma_pos + 1);

    // Parse Board & Features
    chess::Board board(fen);

    if (feature_mode == FeatureMode::Standard ||
        factorized_use_standard_router) {
      FeatureExtractor::fill_input(board, standard_features);
    }
    if (feature_mode == FeatureMode::Factorized) {
      FactorizedFeatureExtractor::fill_input(board, factorized_features);
      pack_factorized_input(factorized_features, packed_factorized_features);
    }

    // Determine side to move
    bool is_white = board.sideToMove() == chess::Color::WHITE;

    // Convert eval to WDL + MateDistance (Side-to-Move perspective)
    WDLOutput wdl = WDLNormalizer::convert(eval_str, is_white);

    // Get raw centipawns for ExpertRouter (STM perspective)
    float eval_cp = WDLNormalizer::to_centipawns(eval_str, is_white);

    // Compute Expert Weights (using centipawns)
    if (feature_mode == FeatureMode::Standard ||
        factorized_use_standard_router) {
      ExpertRouter::compute_weights(standard_features, eval_cp, expert_weights);
    } else {
      ExpertRouter::compute_weights(factorized_features, eval_cp, expert_weights);
    }

    // Deterministic train/val split using Zobrist hash
    // This is 100% stable and uses standard chess hashing
    bool is_val = (board.zobrist() % 100 < 10);

    auto flush_factorized_global = [&](std::vector<PackedFactorizedInput> &buffer,
                                       const std::string &path) {
      if (buffer.size() >= BATCH_CAPACITY) {
        flush_factorized_buffer(path, buffer);
      }
    };

    auto add_factorized_expert_index_entry =
      [&](std::array<std::vector<uint32_t>, ExpertRouter::NUM_EXPERTS>
                &expert_index_buffers,
            std::array<size_t, ExpertRouter::NUM_EXPERTS> &expert_counts,
            const std::array<std::string, ExpertRouter::NUM_EXPERTS>
                &index_paths,
            uint32_t sample_index,
            const ExpertRouter::ExpertWeights &weights) {
          int best_expert = 0;
          float best_score = weights.raw_scores[0];
          for (int i = 1; i < ExpertRouter::NUM_BASE; ++i) {
            if (weights.raw_scores[i] > best_score) {
              best_score = weights.raw_scores[i];
              best_expert = i;
            }
          }

          for (int i = 0; i < ExpertRouter::NUM_BASE; ++i) {
            if (i == best_expert || weights.raw_scores[i] > kBaseExpertThreshold) {
              expert_index_buffers[i].push_back(sample_index);
              expert_counts[i]++;

              if (expert_index_buffers[i].size() >= BATCH_CAPACITY) {
                flush_index_buffer(index_paths[i], expert_index_buffers[i]);
              }
            }
          }

          if (weights.weights[4] > kAuxGateThreshold) {
            expert_index_buffers[4].push_back(sample_index);
            expert_counts[4]++;
            if (expert_index_buffers[4].size() >= BATCH_CAPACITY) {
              flush_index_buffer(index_paths[4], expert_index_buffers[4]);
            }
          }

          if (weights.weights[5] > kAuxGateThreshold) {
            expert_index_buffers[5].push_back(sample_index);
            expert_counts[5]++;
            if (expert_index_buffers[5].size() >= BATCH_CAPACITY) {
              flush_index_buffer(index_paths[5], expert_index_buffers[5]);
            }
          }
        };

    if (is_val) {
      const uint32_t sample_index = static_cast<uint32_t>(val_count);
      if (feature_mode == FeatureMode::Standard) {
        val_feature_writer->add(standard_features);
      } else {
        val_factorized_buffer.push_back(packed_factorized_features);
        flush_factorized_global(val_factorized_buffer,
                                val_dir + "/features.bin");
      }

      val_wdl_writer.add(wdl);
      val_expert_weight_writer.add(expert_weights);

      if (feature_mode == FeatureMode::Standard) {
        val_expert_datasets->add(standard_features, wdl, expert_weights);
      } else if (!factorized_global_only) {
        add_factorized_expert_index_entry(val_factorized_expert_index_buffers,
                                    val_expert_counts,
                                    val_expert_index_paths,
                                    sample_index,
                                    expert_weights);
      }

      val_count++;
    } else {
      const uint32_t sample_index = static_cast<uint32_t>(train_count);
      if (feature_mode == FeatureMode::Standard) {
        train_feature_writer->add(standard_features);
      } else {
        train_factorized_buffer.push_back(packed_factorized_features);
        flush_factorized_global(train_factorized_buffer,
                                train_dir + "/features.bin");
      }

      train_wdl_writer.add(wdl);
      train_expert_weight_writer.add(expert_weights);

      if (feature_mode == FeatureMode::Standard) {
        train_expert_datasets->add(standard_features, wdl, expert_weights);
      } else if (!factorized_global_only) {
        add_factorized_expert_index_entry(train_factorized_expert_index_buffers,
                                    train_expert_counts,
                                    train_expert_index_paths,
                                    sample_index,
                                    expert_weights);
      }

      train_count++;
    }

    processed++;
    pbar.update(processed);
  }
  pbar.complete();

  if (feature_mode == FeatureMode::Factorized) {
    flush_factorized_buffer(train_dir + "/features.bin", train_factorized_buffer);
    flush_factorized_buffer(val_dir + "/features.bin", val_factorized_buffer);

    if (!factorized_global_only) {
      for (int i = 0; i < ExpertRouter::NUM_EXPERTS; ++i) {
        flush_index_buffer(train_expert_index_paths[i],
                           train_factorized_expert_index_buffers[i]);
        flush_index_buffer(val_expert_index_paths[i],
                           val_factorized_expert_index_buffers[i]);
      }
    }
  }

  const double train_pct =
      processed > 0 ? (100.0 * static_cast<double>(train_count) / processed)
                    : 0.0;
  const double val_pct =
      processed > 0 ? (100.0 * static_cast<double>(val_count) / processed)
                    : 0.0;
  const size_t feature_size =
      feature_mode == FeatureMode::Standard ? sizeof(ChessInput)
                        : sizeof(PackedFactorizedInput);

  // Print summary
  std::cout << "\n=== Dataset Summary ===" << std::endl;
  std::cout << "Total processed: " << processed << std::endl;
  std::cout << "Train samples:   " << train_count << " ("
            << train_pct << "%)" << std::endl;
  std::cout << "Val samples:     " << val_count << " ("
            << val_pct << "%)" << std::endl;

  std::cout << "\n--- Train Directory: " << train_dir << " ---" << std::endl;
  std::cout << "Global files:" << std::endl;
  std::cout << "  - features.bin       ("
            << train_count * feature_size / (1024 * 1024) << " MB)"
            << std::endl;
  std::cout << "  - labels.bin         ("
            << train_count * 4 * sizeof(float) / (1024 * 1024) << " MB)"
            << std::endl;
  std::cout << "  - expert_weights.bin ("
            << train_count * 6 * sizeof(float) / (1024 * 1024) << " MB)"
            << std::endl;
  std::cout << "Per-expert files:" << std::endl;
  if (feature_mode == FeatureMode::Standard) {
    train_expert_datasets->print_stats();
  } else if (factorized_global_only) {
    std::cout << "  (skipped in global-only factorized mode)" << std::endl;
  } else {
    std::cout << "  (factorized index mode: expertN_indices.bin)" << std::endl;
    static const char *EXPERT_NAMES[] = {"Tactical",      "Strategic",
                                         "Major Endgame", "Minor Endgame",
                                         "Survivor",      "Killer"};
    size_t total = 0;
    std::cout << "\nExpert Dataset Statistics:" << std::endl;
    for (int i = 0; i < ExpertRouter::NUM_EXPERTS; ++i) {
      std::cout << "  Expert " << i << " (" << EXPERT_NAMES[i]
                << "): " << train_expert_counts[i] << " samples" << std::endl;
      total += train_expert_counts[i];
    }
    std::cout << "  Total entries: " << total << std::endl;
  }

  std::cout << "\n--- Val Directory: " << val_dir << " ---" << std::endl;
  std::cout << "Global files:" << std::endl;
  std::cout << "  - features.bin       ("
            << val_count * feature_size / (1024 * 1024) << " MB)"
            << std::endl;
  std::cout << "  - labels.bin         ("
            << val_count * 4 * sizeof(float) / (1024 * 1024) << " MB)"
            << std::endl;
  std::cout << "  - expert_weights.bin ("
            << val_count * 6 * sizeof(float) / (1024 * 1024) << " MB)"
            << std::endl;
  std::cout << "Per-expert files:" << std::endl;
  if (feature_mode == FeatureMode::Standard) {
    val_expert_datasets->print_stats();
  } else if (factorized_global_only) {
    std::cout << "  (skipped in global-only factorized mode)" << std::endl;
  } else {
    std::cout << "  (factorized index mode: expertN_indices.bin)" << std::endl;
    static const char *EXPERT_NAMES[] = {"Tactical",      "Strategic",
                                         "Major Endgame", "Minor Endgame",
                                         "Survivor",      "Killer"};
    size_t total = 0;
    std::cout << "\nExpert Dataset Statistics:" << std::endl;
    for (int i = 0; i < ExpertRouter::NUM_EXPERTS; ++i) {
      std::cout << "  Expert " << i << " (" << EXPERT_NAMES[i]
                << "): " << val_expert_counts[i] << " samples" << std::endl;
      total += val_expert_counts[i];
    }
    std::cout << "  Total entries: " << total << std::endl;
  }

  return 0;
}
