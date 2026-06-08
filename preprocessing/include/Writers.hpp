/**
 * @file Writers.hpp
 * @brief Utility classes for writing packed dataset features and labels to disk.
 */
#pragma once
#include "ExpertRouter.hpp"     // For ExpertWeights struct
#include "FeatureExtractor.hpp" // For ChessInput struct
#include "WDLNormalizer.hpp"    // For WDLOutput struct
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

struct alignas(64) PackedChessInput {
  // Planes 0-29 are categorical (pieces, moves, threats). 
  // Pack each 64-square plane into a SINGLE 64-bit integer.
  // Size: 30 * 8 bytes = 240 bytes
  uint64_t bitboards[30]; 

  // Planes 30-31 are continuous (King distance), keep as uint8_t
  // Size: 2 * 64 bytes = 128 bytes
  uint8_t continuous_layers[2][64]; 

  // Global features
  // Size: 16 * 4 bytes = 64 bytes
  float global[16]; 
};

class DatasetWriter {
private:
  std::string filename;
  size_t batch_size;

  // Buffer stores the packed structs to save memory
  std::vector<PackedChessInput> buffer;

public:
  DatasetWriter(const std::string &fname, size_t batch_capacity = 1000);
  ~DatasetWriter();
  void add(const ChessInput &features);
  void flush();
};

class LabelWriter {
private:
  std::string filename;
  std::vector<float> buffer;
  size_t batch_size;

public:
  LabelWriter(const std::string &fname, size_t batch_capacity = 1000);
  ~LabelWriter();
  void add(float label);
  void flush();
};

/**
 * WDLWriter - Writes WDL labels during preprocessing
 *
 * Output format: 3 x float32 per position [Win, Draw, Loss]
 * Total: 12 bytes per position
 */
class WDLWriter {
private:
  std::string filename;
  std::vector<float> buffer; // Flattened [w, d, l, m, w, d, l, m, ...]
  size_t batch_size;

public:
  WDLWriter(const std::string &fname, size_t batch_capacity = 1000);
  ~WDLWriter();

  /**
   * Add WDL label for one position
   * @param wdl The WDLOutput struct from WDLNormalizer
   */
  void add(const WDLOutput &wdl);

  void flush();
};

/**
 * ExpertWeightsWriter - Writes expert routing weights during preprocessing
 *
 * Output format: 6 x float32 per position (NUM_EXPERTS weights)
 * Total: 24 bytes per position
 */
class ExpertWeightsWriter {
private:
  std::string filename;
  std::vector<float> buffer; // Flat buffer of all weights
  size_t batch_size;
  static constexpr int NUM_EXPERTS = ExpertRouter::NUM_EXPERTS;

public:
  ExpertWeightsWriter(const std::string &fname, size_t batch_capacity = 1000);
  ~ExpertWeightsWriter();

  /**
   * Add weights for one position
   * @param weights The ExpertWeights struct from ExpertRouter
   */
  void add(const ExpertRouter::ExpertWeights &weights);

  /**
   * Add weights from raw array
   * @param weights Array of NUM_EXPERTS floats
   */
  void add(const float *weights);

  void flush();
};

/**
 * ExpertDatasetWriter - Writes separate feature/label files per expert
 *
 * Creates 6 pairs of files:
 *   expert0_features.bin, expert0_labels.bin  (Tactical)
 *   expert1_features.bin, expert1_labels.bin  (Strategic)
 *   expert2_features.bin, expert2_labels.bin  (Major Endgame)
 *   expert3_features.bin, expert3_labels.bin  (Minor Endgame)
 *   expert4_features.bin, expert4_labels.bin  (Survivor)
 *   expert5_features.bin, expert5_labels.bin  (Killer)
 *
 * Routing:
 *   - Base experts (0-3): Sample goes to argmax(base_weights)
 *   - Aux experts (4-5): Sample goes if gate > 0.1
 */
class ExpertDatasetWriter {
private:
  static constexpr int NUM_EXPERTS = 6;
  static constexpr float AUX_GATE_THRESHOLD = 0.1f;

  std::array<std::unique_ptr<DatasetWriter>, NUM_EXPERTS> feature_writers;
  std::array<std::unique_ptr<WDLWriter>, NUM_EXPERTS> label_writers;
  std::array<size_t, NUM_EXPERTS> counts;

public:
  ExpertDatasetWriter(const std::string &output_dir,
                      size_t batch_capacity = 10000);
  ~ExpertDatasetWriter();

  /**
   * Add sample to appropriate expert(s) based on routing weights
   *
   * @param features The input features
   * @param wdl The WDL labels
   * @param weights The expert routing weights from ExpertRouter
   */
  void add(const ChessInput &features, const WDLOutput &wdl,
           const ExpertRouter::ExpertWeights &weights);

  /**
   * Flush all buffers to disk
   */
  void flush();

  /**
   * Print statistics about samples per expert
   */
  void print_stats() const;
};