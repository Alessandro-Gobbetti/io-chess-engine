#pragma once

/**
 * @file EvalContextMoECache.h
 * @brief Thread-local evaluator for the Factorized Mixture of Experts (MoE) network.
 *
 * Implements an incremental network evaluation context that safely updates its
 * hidden state during the search tree traversal.
 */

#include "IEvaluator.h"
#include "WDLConverter.hpp"
#include "MoECacheModel.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

/**
 * @struct EvalContextMoECacheSharedModel
 * @brief Contains the shared neural network weights.
 */
struct EvalContextMoECacheSharedModel {
  BenchConfig cfg;           ///< Configuration of the MoE architecture.
  SharedMoEWeights weights;  ///< Thread-safe shared weights container.
};

/**
 * @class EvalContextMoECache
 * @brief Thread-local evaluator context utilizing the MoE network.
 *
 * Maintains incremental accumulator state and evaluates positions rapidly.
 */
class EvalContextMoECache : public IEvaluator {
public:
  explicit EvalContextMoECache(const std::string &weightsPath);
  explicit EvalContextMoECache(
      std::shared_ptr<const EvalContextMoECacheSharedModel> sharedModel);

  /**
   * @brief Loads the shared model weights from disk.
   */
  static std::shared_ptr<const EvalContextMoECacheSharedModel>
  loadSharedModel(const std::string &weightsPath);

  float evaluate(const Board &board, int ply = 0) override;
  WDLConverter::WDL evaluateWDL(const Board &board, int ply = 0) override;

  void setAggression(float aggression) override { wdlConverter_.aggression = aggression; }
  void setEvalScale(int base, int weight) override {
    evalScaleBase_ = base;
    evalScaleWeight_ = weight;
  }
  void setEvalNormalization(bool enable) override {
    enableEvalNormalization_ = enable;
  }
  void setIncrementalRebuildInterval(int interval) override {
    rebuildEveryNEvals_ = std::max(0, interval);
  }
  uint64_t getFullRebuilds() const override {
    return totalRebuilds_.load(std::memory_order_relaxed);
  }

private:
  static constexpr uint32_t kMagicWeights = 0x32454F4D; // MOE2
  static constexpr uint32_t kVersion = 1;

  /**
   * @brief Initializes the thread-local context from the pre-loaded shared weights.
   */
  void init_from_shared_model(
      std::shared_ptr<const EvalContextMoECacheSharedModel> shared);

  BenchConfig cfg_{};
  std::shared_ptr<const EvalContextMoECacheSharedModel> sharedModel_{};
  std::array<MoEDoubleAccumulator, 2> models_{};
  
  // Large feature tensor scratch reused across calls to avoid deep-recursion
  // stack growth in alpha-beta.
  FactorizedInput scratchInput_{};

  std::array<FactorizedInput, 2> prevInputByStm_{};
  std::array<bool, 2> hasPrevByStm_{{false, false}};

  WDLConverter wdlConverter_;

  int evalScaleBase_ = 750;
  int evalScaleWeight_ = 25;
  bool enableEvalNormalization_ = true;
  int rebuildEveryNEvals_ = 0;
  std::array<uint32_t, 2> evalsSinceFullByStm_{{0, 0}};
  std::atomic<uint64_t> totalRebuilds_{0};

  /**
   * @brief Reads a 32-bit unsigned integer from the binary weights stream.
   */
  static uint32_t read_u32(std::ifstream &in);
  
  /**
   * @brief Reads a block of float values into a resizing container from the binary stream.
   */
  template <typename FloatContainer>
  static void read_floats(std::ifstream &in, FloatContainer &dst, size_t n) {
    if constexpr (requires(FloatContainer &c, size_t m) { c.resize(m); }) {
      dst.resize(n);
    } else {
      if (n > dst.size())
        throw std::runtime_error("Fixed-size tensor shape mismatch while reading");
    }
    in.read(reinterpret_cast<char *>(dst.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
    if (!in.good())
      throw std::runtime_error("Failed reading float block");
  }
  
  /**
   * @brief Reads a raw block of floats into a contiguous C-style array.
   */
  static void read_floats_raw(std::ifstream &in, float *dst, size_t n);

  /**
   * @brief Converts an integer ID stored in the weights file to an ExpertPoolMode enum.
   */
  static ExpertPoolMode pool_mode_from_code(int code);
  
  /**
   * @brief Copies and transposes a matrix tensor (used to align weights for faster cache hits during inference).
   */
  template <typename SrcContainer, typename DstContainer>
  static void transpose_copy(const SrcContainer &src, DstContainer &dst,
                             int rows, int cols) {
    const size_t n = static_cast<size_t>(rows) * cols;
    if constexpr (requires(DstContainer &c, size_t m) { c.resize(m); }) {
      dst.resize(n);
    } else {
      if (dst.size() != n)
        throw std::runtime_error("Fixed-size tensor shape mismatch while transposing");
    }
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        dst[(size_t)c * rows + r] = src[(size_t)r * cols + c];
      }
    }
  }
  
  /**
   * @brief Loads neural network weights from a binary file into a shared weights container.
   */
  static void load_weights_into_target(const std::string &weightsPath,
                                       BenchConfig &cfg,
                                       SharedMoEWeights &weights);

  /**
   * @brief Loads neural network weights directly into the thread-local model (legacy usage).
   */
  void load_weights_into_model(const std::string &weightsPath);
};
