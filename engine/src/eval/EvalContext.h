#pragma once

#include "IEvaluator.h"
#include "WDLConverter.hpp"
#include "EvalUtils.h"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <string>

// ============================================================================
// EvalContext - Per-thread context using SHARED session
// ============================================================================
class EvalContext : public IEvaluator {
private:
  // SHARED session (not owned - managed by Evaluator)
  Ort::Session* session_;
  Ort::AllocatorWithDefaultOptions allocator_;

  // Per-thread aligned buffers for GPU compatibility
  AlignedBuffer<uint8_t> layersBuffer_;  // [32 * 64] uint8
  AlignedBuffer<float> globalBuffer_;     // [15] float

  // === OPTIMIZATION: Cached names and memory info (avoid per-call allocations) ===
  std::vector<const char*> inputNames_;
  std::vector<const char*> outputNames_;
  std::vector<Ort::AllocatedStringPtr> inputNameStorage_;
  std::vector<Ort::AllocatedStringPtr> outputNameStorage_;
  Ort::MemoryInfo memoryInfo_;
  
  // Pre-allocated tensor shapes (avoid stack allocations)
  std::vector<int64_t> layersShape_;
  std::vector<int64_t> globalShape_;
  
  // Cached WDL converter
  WDLConverter wdlConverter_;

public:
  // Constructor taking shared session pointer
  EvalContext(Ort::Session* sharedSession);

  // Legacy constructor for backward compatibility (creates own session)
  EvalContext(Ort::Env &env, const std::string &modelPath, int numThreads);

  // IEvaluator interface
  float evaluate(const Board &board) override;
  void fillFeatures(const Board &board, float *buffer);
  
  // Helper to calculate WDL directly (if supported)
  // For standard EvalContext, we can calculate from CP or if model outputs WDL
  // Current model outputs WDL (4 floats), so we can expose that.
  WDLConverter::WDL evaluateWDL(const Board &board);

  // Set aggression parameter for WDL conversion
  void setAggression(float aggression) override {
    wdlConverter_.aggression = aggression;
  }

private:
  // Owned session for legacy constructor
  std::unique_ptr<Ort::Session> ownedSession_;
  
  // Helper to initialize cached names
  void initCachedNames();
};
