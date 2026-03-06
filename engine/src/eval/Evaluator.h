#pragma once
// Evaluator.h - ONNX Runtime-based evaluation context
// Standard CPU inference for Chess Engine

#include "IEvaluator.h"
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// ============================================================================
// Evaluator - Model manager with SHARED sessions
// ============================================================================
class Evaluator {
private:
  Ort::Env env_;
  std::string modelPath_;
  int numThreads_;
  bool isMoE_;
  bool useGPU_;

  // SHARED sessions (created once, used by all thread contexts)
  std::unique_ptr<Ort::Session> sharedSession_;
  std::unique_ptr<Ort::Session> sharedBackboneSession_;
  std::vector<std::unique_ptr<Ort::Session>> sharedExpertSessions_;

public:
  Evaluator(const std::string &modelPath, int numThreads, bool useGPU = false);

  // Create lightweight thread context from shared sessions
  std::unique_ptr<IEvaluator> createThreadContext();
  
  bool isLoaded() const { return sharedSession_ != nullptr || sharedBackboneSession_ != nullptr; }
  bool isMoE() const { return isMoE_; }
};