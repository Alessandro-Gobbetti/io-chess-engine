// Evaluator.cpp - ONNX Runtime-based evaluation context
// 
// LAZY SMP OPTIMIZATION: Session sharing
// This implementation supports two modes:
// 1. Shared session mode (preferred for Lazy SMP): Multiple thread contexts
//    share a single ONNX session, avoiding memory explosion and thread contention
// 2. Legacy mode: Each context owns its own session (backward compatibility)

#include "Evaluator.h"
#include "EvalContext.h"
#include "EvalContextMoE.h"
#include "EvalUtils.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

// ============================================================================
// Evaluator - Model manager with SHARED sessions for Lazy SMP
// ============================================================================
//
// CRITICAL OPTIMIZATION: Load model ONCE and share across all search threads.
// Before: N threads × 100MB model = N×100MB memory + N×evalThreads ONNX threads
// After:  1 shared model = 100MB memory + evalThreads ONNX threads
//
// This eliminates:
// 1. Memory explosion from duplicate model weights
// 2. Thread pool explosion causing CPU over-subscription
// 3. Cache thrashing from multiple weight copies competing for L2/L3

// Helper to create Env with specific settings for WASM
static Ort::Env createEnv() {
  const std::string name = "ChessEngine";
#ifdef __EMSCRIPTEN__
  OrtEnv* env_ptr = nullptr;
  OrtThreadingOptions* tp_options = nullptr;
  
  // Use C API to create Env with Global Thread Pools (required for WASM threads)
  const OrtApi& api = Ort::GetApi();
  
  if (auto status = api.CreateThreadingOptions(&tp_options)) {
      api.ReleaseStatus(status);
      std::cerr << "Failed to create threading options" << std::endl;
      std::abort();
  }
  
  if (auto status = api.CreateEnvWithGlobalThreadPools(
          ORT_LOGGING_LEVEL_WARNING, name.c_str(), tp_options, &env_ptr)) {
      api.ReleaseStatus(status);
      api.ReleaseThreadingOptions(tp_options);
      std::cerr << "Failed to create global thread pool env" << std::endl;
      std::abort();
  }
  
  api.ReleaseThreadingOptions(tp_options);
  return Ort::Env(env_ptr);
#else
  return Ort::Env(ORT_LOGGING_LEVEL_WARNING, name.c_str());
#endif
}

Evaluator::Evaluator(const std::string &modelPath, int numThreads, bool useGPU)
    : env_(createEnv()),
      modelPath_(modelPath),
      numThreads_(numThreads),
      isMoE_(false),
      useGPU_(useGPU) {

  // Check if it's a directory (MoE) or single file
  fs::path dir(modelPath);
  if (fs::exists(dir / "backbone.onnx")) {
      isMoE_ = true;
  } else {
      isMoE_ = fs::is_directory(modelPath);
  }

  // Determine if GPU should be used
  // useGPU parameter explicitly specifies user preference
  if (useGPU) {
    // User requested GPU - try to use it but fall back to CPU if unavailable
    std::cout << "info string ONNX: Attempting GPU (CUDA) execution" << std::endl;
  } else {
    // User requested CPU - don't attempt GPU
    std::cout << "info string ONNX: Using CPU execution provider" << std::endl;
  }

  // Create session options for shared session
  auto sessionOptions = createSessionOptions(numThreads, useGPU);
  
  // Try to add GPU provider if requested
  if (useGPU) {
    try {
      // Add CUDA execution provider with fallback to CPU
      OrtCUDAProviderOptions cudaOptions;
      cudaOptions.device_id = 0;
      sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
      // Note: CPU is the default fallback, no need to explicitly add it
      std::cout << "info string ONNX: CUDA execution provider enabled" << std::endl;
    } catch (const Ort::Exception &e) {
      std::cout << "info string ONNX: CUDA not available, falling back to CPU: " << e.what() << std::endl;
      useGPU_ = false;
    }
  } else {
    // CPU is default - no need to explicitly add provider
    useGPU_ = false;
  }

  if (isMoE_) {
    // MoE: Load shared backbone and expert sessions
    fs::path dir(modelPath);
    
    auto backbonePath = dir / "backbone.onnx";
    std::cout << "info string ONNX loading SHARED backbone: " << backbonePath << std::endl;
    sharedBackboneSession_ = std::make_unique<Ort::Session>(
        env_, backbonePath.string().c_str(), sessionOptions);
    
    for (int i = 0; i < 4; ++i) {
      auto expertPath = dir / ("expert" + std::to_string(i) + ".onnx");
      std::cout << "info string ONNX loading SHARED expert: " << expertPath << std::endl;
      sharedExpertSessions_.push_back(std::make_unique<Ort::Session>(
          env_, expertPath.string().c_str(), sessionOptions));
    }
    
    std::cout << "info string MoE model loaded with SHARED sessions (" << (useGPU_ ? "GPU" : "CPU") << ")" << std::endl;
  } else {
    // Standard model: Load single shared session
    std::cout << "info string ONNX loading SHARED model: " << modelPath << std::endl;
    sharedSession_ = std::make_unique<Ort::Session>(
        env_, modelPath.c_str(), sessionOptions);
  }

  std::cout << "info string ONNX Runtime: Created shared session(s) with " 
            << numThreads << " eval threads" << std::endl;
  std::cout << "info string LAZY SMP: All search threads will share this session" << std::endl;
}

std::unique_ptr<IEvaluator> Evaluator::createThreadContext() {
  if (isMoE_) {
    std::vector<Ort::Session*> expertPtrs;
    for (auto& expert : sharedExpertSessions_) {
      expertPtrs.push_back(expert.get());
    }
    return std::make_unique<EvalContextMoE>(
        sharedBackboneSession_.get(), expertPtrs);
  } else {
    return std::make_unique<EvalContext>(sharedSession_.get());
  }
}
