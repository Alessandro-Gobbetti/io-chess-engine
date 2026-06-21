/**
 * @file Evaluator.cpp
 * @brief Runtime evaluator manager implementation.
 *
 * Handles the instantiation of thread-local evaluation contexts, loading shared neural 
 * network weights, and routing evaluation requests during the game.
 * @ingroup engine
 */
// Evaluator.cpp - runtime evaluator manager
// Native incremental MoE cache only.

#include "Evaluator.h"
#include "EvalContextMoECache.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

Evaluator::Evaluator(const std::string &modelPath, int numThreads, bool useGPU)
    : modelPath_(modelPath),
      nativeWeightsPath_(""),
      numThreads_(numThreads),
      isMoECache_(false),
      useGPU_(useGPU) {
  (void)numThreads_;
  (void)useGPU_;

  fs::path dir(modelPath);
  if (fs::exists(dir / "native_weights.bin")) {
    isMoECache_ = true;
    nativeWeightsPath_ = (dir / "native_weights.bin").string();
    std::cout << "info string Native incremental MoE cache evaluator enabled" << std::endl;
    std::cout << "info string Loading native weights (shared): " << nativeWeightsPath_ << std::endl;
    sharedMoECacheModel_ = EvalContextMoECache::loadSharedModel(nativeWeightsPath_);
    return;
  }

  throw std::runtime_error(
      "This engine build supports native model bundles only. "
      "Provide a directory containing native_weights.bin.");
}

std::unique_ptr<IEvaluator> Evaluator::createThreadContext() {
  if (isMoECache_) {
    return std::make_unique<EvalContextMoECache>(sharedMoECacheModel_);
  }
  throw std::runtime_error("Evaluator is not loaded");
}
