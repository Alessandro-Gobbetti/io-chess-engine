#include "EvalContext.h"
#include <FeatureExtractor.hpp>
#include <cstring>

// Helper to initialize cached names (called once in constructor)
void EvalContext::initCachedNames() {
  // Cache input names
  auto inputCount = session_->GetInputCount();
  inputNameStorage_.reserve(inputCount);
  inputNames_.reserve(inputCount);
  for (size_t i = 0; i < inputCount; ++i) {
    inputNameStorage_.push_back(session_->GetInputNameAllocated(i, allocator_));
    inputNames_.push_back(inputNameStorage_.back().get());
  }
  
  // Cache output names
  auto outputCount = session_->GetOutputCount();
  outputNameStorage_.reserve(outputCount);
  outputNames_.reserve(outputCount);
  for (size_t i = 0; i < outputCount; ++i) {
    outputNameStorage_.push_back(session_->GetOutputNameAllocated(i, allocator_));
    outputNames_.push_back(outputNameStorage_.back().get());
  }
}

// NEW: Constructor for shared session mode (Lazy SMP optimized)
EvalContext::EvalContext(Ort::Session* sharedSession)
    : session_(sharedSession), ownedSession_(nullptr),
      layersBuffer_(32 * 64), globalBuffer_(15),
      memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      layersShape_({1, 32, 8, 8}), globalShape_({1, 15}) {
  // Initialize cached names for this session
  initCachedNames();
}

// LEGACY: Constructor that creates its own session (backward compatibility)
EvalContext::EvalContext(Ort::Env &env, const std::string &modelPath,
                                   int numThreads)
    : session_(nullptr), ownedSession_(nullptr),
      layersBuffer_(32 * 64), globalBuffer_(15),
      memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      layersShape_({1, 32, 8, 8}), globalShape_({1, 15}) {

  auto sessionOptions = createSessionOptions(numThreads);

  // Create owned session
  ownedSession_ = std::make_unique<Ort::Session>(env, modelPath.c_str(), sessionOptions);
  session_ = ownedSession_.get();
  
  // Initialize cached names for this session
  initCachedNames();
}

void EvalContext::fillFeatures(const Board &board, float *buffer) {
  ChessInput features{};
  FeatureExtractor::fill_input(board, features);

  // Convert uint8 layers to float
  // features.layers is uint8_t[32][64]
  const uint8_t* layerData = reinterpret_cast<const uint8_t*>(features.layers);
  for (int i = 0; i < 32 * 64; ++i) {
    buffer[i] = static_cast<float>(layerData[i]) / 255.0f;
  }
}

float EvalContext::evaluate(const Board &board) {
  // Extract features into stack buffer, then copy to aligned memory
  ChessInput features{};
  FeatureExtractor::fill_input(board, features);
  
  // Copy to aligned buffers for GPU compatibility
  std::memcpy(layersBuffer_.data(), features.layers, 32 * 64);
  std::memcpy(globalBuffer_.data(), features.global, 15 * sizeof(float));

  // Create input tensors using pre-allocated shapes and memory info
  std::vector<Ort::Value> inputTensors;
  inputTensors.reserve(2);

  // layers input (uint8) from aligned buffer
  inputTensors.push_back(Ort::Value::CreateTensor<uint8_t>(
      memoryInfo_, layersBuffer_.data(), 32 * 64, layersShape_.data(), 4));

  // global_features input (float) from aligned buffer
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo_, globalBuffer_.data(), 15, globalShape_.data(), 2));

  // Run inference using cached names (avoid per-call allocations)
  auto outputTensors = session_->Run(Ort::RunOptions{nullptr},
                                     inputNames_.data(), inputTensors.data(), inputNames_.size(),
                                     outputNames_.data(), outputNames_.size());

  // Get output
  float *output = outputTensors[0].GetTensorMutableData<float>();

  // Convert WDL to centipawns using cached converter
  return wdlConverter_.convert(output[0], output[1], output[2], output[3]);
}

WDLConverter::WDL EvalContext::evaluateWDL(const Board &board) {
  // Same as evaluate but returns WDL struct
  ChessInput features{};
  FeatureExtractor::fill_input(board, features);
  
  std::memcpy(layersBuffer_.data(), features.layers, 32 * 64);
  std::memcpy(globalBuffer_.data(), features.global, 15 * sizeof(float));

  std::vector<Ort::Value> inputTensors;
  inputTensors.reserve(2);
  inputTensors.push_back(Ort::Value::CreateTensor<uint8_t>(
      memoryInfo_, layersBuffer_.data(), 32 * 64, layersShape_.data(), 4));
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo_, globalBuffer_.data(), 15, globalShape_.data(), 2));

  auto outputTensors = session_->Run(Ort::RunOptions{nullptr},
                                     inputNames_.data(), inputTensors.data(), inputNames_.size(),
                                     outputNames_.data(), outputNames_.size());

  float *output = outputTensors[0].GetTensorMutableData<float>();
  
  WDLConverter::WDL wdl;
  wdl.win = output[0];
  wdl.draw = output[1];
  wdl.loss = output[2];
  wdl.mate = output[3];
  
  return wdl;
}
