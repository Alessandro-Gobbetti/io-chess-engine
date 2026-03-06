#pragma once

#include "IEvaluator.h"
#include "WDLConverter.hpp"
#include "EvalUtils.h"
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>
#include <string>

// ============================================================================
// EvalContextMoE - Per-thread MoE context using SHARED sessions
// ============================================================================
class EvalContextMoE : public IEvaluator {
private:
  // SHARED sessions (not owned - managed by Evaluator)
  Ort::Session* backboneSession_;
  std::vector<Ort::Session*> expertSessions_;
  Ort::AllocatorWithDefaultOptions allocator_;

  // Per-thread aligned buffers for GPU compatibility
  AlignedBuffer<uint8_t> layersBuffer_;  // [32 * 64] uint8
  AlignedBuffer<float> globalBuffer_;     // [15] float

  // === OPTIMIZATION: Cached names and memory info (avoid per-call allocations) ===
  std::vector<const char*> backboneInputNames_;
  std::vector<const char*> backboneOutputNames_;
  std::vector<Ort::AllocatedStringPtr> backboneInputNameStorage_;
  std::vector<Ort::AllocatedStringPtr> backboneOutputNameStorage_;
  
  // Per-expert cached names
  std::vector<std::vector<const char*>> expertInputNames_;
  std::vector<std::vector<const char*>> expertOutputNames_;
  std::vector<std::vector<Ort::AllocatedStringPtr>> expertInputNameStorage_;
  std::vector<std::vector<Ort::AllocatedStringPtr>> expertOutputNameStorage_;
  
  Ort::MemoryInfo memoryInfo_;
  std::vector<int64_t> layersShape_;
  std::vector<int64_t> globalShape_;
  WDLConverter wdlConverter_;

  // === OPTIMIZATION: IO Bindings for zero-copy execution ===
  std::unique_ptr<Ort::IoBinding> backboneBinding_;
  std::vector<std::unique_ptr<Ort::IoBinding>> expertBindings_;
  
  // Persistent buffers for outputs
  AlignedBuffer<float> backboneOutputBuffer_;
  AlignedBuffer<float> expertOutputBuffer_; // FIXED: Single shared buffer instead of vector
  
  std::vector<int64_t> backboneOutputShape_;
  std::vector<int64_t> expertOutputShape_;

public:
  // Constructor taking shared session pointers
  EvalContextMoE(Ort::Session* sharedBackbone, 
                     const std::vector<Ort::Session*>& sharedExperts);

  // Legacy constructor for backward compatibility (creates own sessions)
  EvalContextMoE(Ort::Env &env, const std::string &modelDir, int numThreads);

  float evaluate(const Board &board) override;
  WDLConverter::WDL evaluateWDL(const Board &board) override;
  void fillFeatures(const Board &board, float *buffer);

  // === Component-level timing for benchmarking ===
  struct ComponentTimings {
    double featureExtractUs = 0;  // Feature extraction (FeatureExtractor + ExpertRouter)
    double backboneUs = 0;       // Backbone ONNX session run
    double expert0Us = 0;        // Expert 0 session run
    double expert1Us = 0;        // Expert 1 session run
    double expert2Us = 0;        // Expert 2 session run
    double expert3Us = 0;        // Expert 3 session run
    double wdlConvertUs = 0;     // WDL→CP conversion + material scaling
    double totalUs = 0;          // Total evaluate() time
    int topExpert0 = -1;         // Index of top expert
    int topExpert1 = -1;         // Index of second expert
    float topProb0 = 0;          // Top expert probability
    float topProb1 = 0;          // Second expert probability
  };
  ComponentTimings benchmarkComponents(const Board &board);

  // Set aggression parameter for WDL conversion
  void setAggression(float aggression) override {
    wdlConverter_.aggression = aggression;
  }
  
  void setEvalScale(int base, int weight) override {
    evalScaleBase_ = base;
    evalScaleWeight_ = weight;
  }

  void setEvalNormalization(bool enable) override {
    enableEvalNormalization_ = enable;
  }

private:
  int evalScaleBase_ = 750;
  int evalScaleWeight_ = 25;
  bool enableEvalNormalization_ = true;
  
  // Owned sessions for legacy constructor
  std::unique_ptr<Ort::Session> ownedBackboneSession_;
  std::vector<std::unique_ptr<Ort::Session>> ownedExpertSessions_;
  
  // Helper to initialize cached names
  void initCachedNames();
  
  // Helper to setup IO bindings once
  void setupBindings();
};
