#pragma once

/**
 * @file Evaluator.h
 * @brief Runtime evaluator manager.
 *
 * Supports native incremental MoE cache bundles for engine runtime,
 * managing the shared models and creating thread-local contexts.
 */

#include "IEvaluator.h"
#include <memory>
#include <string>

struct EvalContextMoECacheSharedModel;

/**
 * @class Evaluator
 * @brief Model manager that shares native weights across threads.
 *
 * It is responsible for loading the neural network architecture (or falling
 * back to simple evaluation) and instantiating thread-local `IEvaluator`
 * contexts that share the read-only weights to save memory.
 */
class Evaluator {
private:
  std::string modelPath_;
  std::string nativeWeightsPath_;
  int numThreads_;
  bool isMoECache_;
  bool useGPU_;

  std::shared_ptr<const EvalContextMoECacheSharedModel> sharedMoECacheModel_; ///< Shared native model used by all thread contexts.

public:
  /**
   * @brief Constructs an Evaluator manager.
   * 
   * @param modelPath Path to the network weights file.
   * @param numThreads Number of threads that will request contexts.
   * @param useGPU Whether to accelerate evaluation using the GPU (if supported).
   */
  Evaluator(const std::string &modelPath, int numThreads, bool useGPU = false);

  /**
   * @brief Creates a lightweight thread context from the shared model.
   * 
   * @return A unique pointer to an IEvaluator instance for a specific thread.
   */
  std::unique_ptr<IEvaluator> createThreadContext();
  
  bool isLoaded() const { return isMoECache_; }
  bool isMoE() const { return false; }
  bool isMoECache() const { return isMoECache_; }
};