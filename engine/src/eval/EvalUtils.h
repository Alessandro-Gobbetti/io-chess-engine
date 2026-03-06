#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <onnxruntime_cxx_api.h>

// ============================================================================
// AlignedBuffer - RAII wrapper for aligned memory allocation
// ============================================================================
// GPU execution providers may require specific memory alignment (4096-byte for OpenCL)
template<typename T, size_t Alignment = 4096>
class AlignedBuffer {
public:
  AlignedBuffer() : data_(nullptr), size_(0) {}
  
  // FIX: Initialize data_(nullptr) to prevent free() on garbage if count is 0
  explicit AlignedBuffer(size_t count) : data_(nullptr), size_(count) {
    if (count > 0) {
      size_t bytes = count * sizeof(T);
      // std::aligned_alloc requires size to be a multiple of alignment
      size_t alignedBytes = ((bytes + Alignment - 1) / Alignment) * Alignment;
      data_ = static_cast<T*>(std::aligned_alloc(Alignment, alignedBytes));
      if (!data_) throw std::bad_alloc();
    }
  }
  
  ~AlignedBuffer() { if (data_) std::free(data_); }
  
  // Move only
  AlignedBuffer(AlignedBuffer&& other) noexcept 
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  
  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
      if (data_) std::free(data_);
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }
  
  // No copy
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  
  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  
private:
  T* data_;
  size_t size_;
};

// ============================================================================
// Helper: Configure ONNX session options
// ============================================================================
inline Ort::SessionOptions createSessionOptions(int numThreads, bool useGPU = false) {
  Ort::SessionOptions sessionOptions;
  
  // Graph optimization
#ifdef __EMSCRIPTEN__
  // Enable all for WASM (requires custom build with FusedConv/FusedGemm)
  sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#else
  // Enable all for native build (includes FusedConv, etc.)
  sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#endif
  
  if (useGPU) {
    // === GPU OPTIMIZATION ===
    // GPU handles its own threading, keep CPU threads minimal
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);
    std::cout << "info string ONNX: Configured for GPU execution" << std::endl;
#ifdef __EMSCRIPTEN__
  } else {
    // === WASM OPTIMIZATION ===
    // WASM SharedArrayBuffer atomics have high overhead. For small neural networks,
    // single-threaded ONNX is actually faster than multi-threaded due to sync costs.
    // See: https://emscripten.org/docs/porting/pthreads.html
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.AddConfigEntry("session.intra_op.allow_spinning", "0");
    sessionOptions.AddConfigEntry("session.inter_op.allow_spinning", "0");
    std::cout << "info string ONNX: WASM single-threaded mode (optimized for low overhead)" << std::endl;
  }
#else
  } else if (numThreads == 1) {
    // === SINGLE-THREADED CPU OPTIMIZATION ===
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    sessionOptions.AddConfigEntry("session.intra_op.allow_spinning", "0");
    sessionOptions.AddConfigEntry("session.inter_op.allow_spinning", "0");
  } else {
    // === MULTI-THREADED (LAZY SMP) OPTIMIZATION ===
    // We respect the requested numThreads (EvalThreads).
    // UCI.cpp's smart logic ensures we don't oversubscribe by default.
    // If the user explicitly sets EvalThreads > 1, we trust they have enough cores.
    sessionOptions.SetIntraOpNumThreads(numThreads);
    sessionOptions.SetInterOpNumThreads(1);
    // For feed-forward networks (Chess NN), there is little INTER-op parallelism.
    // Sequential execution avoids the overhead of the parallel executor while still
    // allowing INTRA-op parallelism (multithreaded matrix mul).
    sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    // Spinning helps latency when we have spare cores
    sessionOptions.AddConfigEntry("session.intra_op.allow_spinning", "1");
    sessionOptions.AddConfigEntry("session.inter_op.allow_spinning", "1");
  }
#endif
  
  // Memory optimizations - enable pattern matching and arena allocators
  sessionOptions.AddConfigEntry("session.use_env_allocators", "1");
  
  // CPU-specific optimizations for faster inference
  if (!useGPU) {
    sessionOptions.AddConfigEntry("session.use_ort_model_bytes_directly", "1");
    sessionOptions.AddConfigEntry("session.use_ort_model_bytes_for_initializers", "1");
    sessionOptions.AddConfigEntry("session.disable_cpu_ep_fallback", "0");
  }
  
  return sessionOptions;
}
