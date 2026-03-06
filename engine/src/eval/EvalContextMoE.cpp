#include "EvalContextMoE.h"
#include <FeatureExtractor.hpp>
#include <ExpertRouter.hpp>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

// Helper to initialize cached names (called once in constructor)
void EvalContextMoE::initCachedNames() {
  // Cache backbone input names
  auto backboneInputCount = backboneSession_->GetInputCount();
  backboneInputNameStorage_.reserve(backboneInputCount);
  backboneInputNames_.reserve(backboneInputCount);
  for (size_t i = 0; i < backboneInputCount; ++i) {
    backboneInputNameStorage_.push_back(backboneSession_->GetInputNameAllocated(i, allocator_));
    backboneInputNames_.push_back(backboneInputNameStorage_.back().get());
  }
  
  // Cache backbone output names
  auto backboneOutputCount = backboneSession_->GetOutputCount();
  backboneOutputNameStorage_.reserve(backboneOutputCount);
  backboneOutputNames_.reserve(backboneOutputCount);
  for (size_t i = 0; i < backboneOutputCount; ++i) {
    backboneOutputNameStorage_.push_back(backboneSession_->GetOutputNameAllocated(i, allocator_));
    backboneOutputNames_.push_back(backboneOutputNameStorage_.back().get());
  }
  
  // Cache expert names for each expert
  expertInputNames_.resize(expertSessions_.size());
  expertOutputNames_.resize(expertSessions_.size());
  expertInputNameStorage_.resize(expertSessions_.size());
  expertOutputNameStorage_.resize(expertSessions_.size());
  
  for (size_t e = 0; e < expertSessions_.size(); ++e) {
    auto inputCount = expertSessions_[e]->GetInputCount();
    expertInputNameStorage_[e].reserve(inputCount);
    expertInputNames_[e].reserve(inputCount);
    for (size_t i = 0; i < inputCount; ++i) {
      expertInputNameStorage_[e].push_back(expertSessions_[e]->GetInputNameAllocated(i, allocator_));
      expertInputNames_[e].push_back(expertInputNameStorage_[e].back().get());
    }
    
    auto outputCount = expertSessions_[e]->GetOutputCount();
    expertOutputNameStorage_[e].reserve(outputCount);
    expertOutputNames_[e].reserve(outputCount);
    for (size_t i = 0; i < outputCount; ++i) {
      expertOutputNameStorage_[e].push_back(expertSessions_[e]->GetOutputNameAllocated(i, allocator_));
      expertOutputNames_[e].push_back(expertOutputNameStorage_[e].back().get());
    }
  }
}

// Helper to setup IO bindings for zero-copy execution
void EvalContextMoE::setupBindings() {
  // 1. Determine backbone output size from model metadata
  auto backboneOutputInfo = backboneSession_->GetOutputTypeInfo(0);
  auto backboneOutputTensorInfo = backboneOutputInfo.GetTensorTypeAndShapeInfo();
  auto backboneOutShape = backboneOutputTensorInfo.GetShape();
  size_t backboneOutputSize = backboneOutputTensorInfo.GetElementCount();
  
  // Store shape for later use
  backboneOutputShape_ = std::vector<int64_t>(backboneOutShape.begin(), backboneOutShape.end());
  
  // Allocate persistent backbone output buffer
  backboneOutputBuffer_ = AlignedBuffer<float>(backboneOutputSize);
  
  // Expert output is always [1, 4] for WDL
  expertOutputShape_ = {1, 4};
  expertOutputBuffer_ = AlignedBuffer<float>(4);
  
  // 2. Setup Backbone Binding
  backboneBinding_ = std::make_unique<Ort::IoBinding>(*backboneSession_);
  
  // Bind inputs to existing member buffers
  backboneBinding_->BindInput(backboneInputNames_[0],
      Ort::Value::CreateTensor<uint8_t>(memoryInfo_, layersBuffer_.data(), 
                                         32 * 64, layersShape_.data(), 4));
  
  backboneBinding_->BindInput(backboneInputNames_[1],
      Ort::Value::CreateTensor<float>(memoryInfo_, globalBuffer_.data(), 
                                       15, globalShape_.data(), 2));
  
  // Bind output to persistent buffer for zero-copy routing to experts
  backboneBinding_->BindOutput(backboneOutputNames_[0],
      Ort::Value::CreateTensor<float>(memoryInfo_, backboneOutputBuffer_.data(),
                                       backboneOutputSize, backboneOutputShape_.data(),
                                       backboneOutputShape_.size()));
  
  // 3. Setup Expert Bindings (one for each expert)
  expertBindings_.reserve(expertSessions_.size());
  for (size_t i = 0; i < expertSessions_.size(); ++i) {
    expertBindings_.push_back(std::make_unique<Ort::IoBinding>(*expertSessions_[i]));
    
    // Expert input binds to backbone output buffer (zero-copy!)
    expertBindings_[i]->BindInput(expertInputNames_[i][0],
        Ort::Value::CreateTensor<float>(memoryInfo_, backboneOutputBuffer_.data(),
                                         backboneOutputSize, backboneOutputShape_.data(),
                                         backboneOutputShape_.size()));
    
    // Bind expert output to persistent buffer
    expertBindings_[i]->BindOutput(expertOutputNames_[i][0],
        Ort::Value::CreateTensor<float>(memoryInfo_, expertOutputBuffer_.data(),
                                         4, expertOutputShape_.data(), 2));
  }
}

// NEW: Constructor for shared session mode (Lazy SMP optimized)
EvalContextMoE::EvalContextMoE(Ort::Session* sharedBackbone, 
                                       const std::vector<Ort::Session*>& sharedExperts)
    : backboneSession_(sharedBackbone), expertSessions_(sharedExperts),
      ownedBackboneSession_(nullptr),
      layersBuffer_(32 * 64), globalBuffer_(15),
      memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      layersShape_({1, 32, 8, 8}), globalShape_({1, 15}),
      backboneOutputBuffer_(0), expertOutputBuffer_(4) { // Will resize in setupBindings
  // Initialize cached names for all sessions
  initCachedNames();
  // Setup IO bindings for zero-copy execution
  setupBindings();
}

// LEGACY: Constructor that creates its own sessions (backward compatibility)
EvalContextMoE::EvalContextMoE(Ort::Env &env, const std::string &modelDir,
                                         int numThreads)
    : backboneSession_(nullptr),
      layersBuffer_(32 * 64), globalBuffer_(15),
      memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      layersShape_({1, 32, 8, 8}), globalShape_({1, 15}) {

  auto sessionOptions = createSessionOptions(numThreads);

  fs::path dir(modelDir);

  // Load owned backbone
  auto backbonePath = dir / "backbone.onnx";
  // std::cout << "info string ONNX loading backbone: " << backbonePath << std::endl;
  ownedBackboneSession_ = std::make_unique<Ort::Session>(env, backbonePath.string().c_str(), sessionOptions);
  backboneSession_ = ownedBackboneSession_.get();

  // Load owned experts
  for (int i = 0; i < 4; ++i) {
    auto expertPath = dir / ("expert" + std::to_string(i) + ".onnx");
    // std::cout << "info string ONNX loading expert: " << expertPath << std::endl;
    ownedExpertSessions_.push_back(std::make_unique<Ort::Session>(
        env, expertPath.string().c_str(), sessionOptions));
    expertSessions_.push_back(ownedExpertSessions_.back().get());
  }
  
  // Initialize cached names for all sessions
  initCachedNames();
  // Setup IO bindings for zero-copy execution
  setupBindings();
}

void EvalContextMoE::fillFeatures(const Board &board, float *buffer) {
  ChessInput features{};
  FeatureExtractor::fill_input(board, features);

  // features.layers is uint8_t[32][64]
  const uint8_t* layerData = reinterpret_cast<const uint8_t*>(features.layers);
  for (int i = 0; i < 32 * 64; ++i) {
    buffer[i] = static_cast<float>(layerData[i]) / 255.0f;
  }
}

float EvalContextMoE::evaluate(const Board &board) {
  // 1. Extract features and fill input buffers
  ChessInput features{};
  FeatureExtractor::fill_input(board, features);
  
  // Copy to aligned buffers (these are bound to the backbone inputs)
  std::memcpy(layersBuffer_.data(), features.layers, 32 * 64);
  std::memcpy(globalBuffer_.data(), features.global, 15 * sizeof(float));

  // 2. Compute routing weights client-side
  ExpertRouter::ExpertWeights weights;
  ExpertRouter::compute_weights(features, 0.0f, weights);

  // Get Top-2 Experts
  int top_idx[2];
  float top_probs[2];
  ExpertRouter::get_top_k(weights, 2, top_idx, top_probs);

  // Optimization: If primary expert is very confident, skip second
  int k_limit = 2;
  if (top_probs[0] > 0.90f) {
    k_limit = 1;
    top_probs[0] = 1.0f;
  }

  // 3. Run backbone using IO binding (zero allocations!)
  // SynchronizeInputs() handles copying from CPU buffer to device if using GPU
  backboneBinding_->SynchronizeInputs();
  backboneSession_->Run(Ort::RunOptions{nullptr}, *backboneBinding_);
  backboneBinding_->SynchronizeOutputs();
  
  // Backbone output is now in backboneOutputBuffer_, which is also bound
  // as the input to all expert bindings (zero-copy routing!)

  // 4. Run selected experts and accumulate WDL
  float final_wdl[4] = {0.0f};

  for (int k = 0; k < k_limit; ++k) {
    int expertIdx = top_idx[k];
    float prob = top_probs[k];

    // Run expert using IO binding (zero allocations, zero-copy input!)
    expertBindings_[expertIdx]->SynchronizeInputs();
    expertSessions_[expertIdx]->Run(Ort::RunOptions{nullptr}, *expertBindings_[expertIdx]);
    expertBindings_[expertIdx]->SynchronizeOutputs();
    
    // Expert output is in expertOutputBuffer_
    const float* expertOut = expertOutputBuffer_.data();

    // Weighted sum of WDL outputs
    final_wdl[0] += expertOut[0] * prob; // Win
    final_wdl[1] += expertOut[1] * prob; // Draw
    final_wdl[2] += expertOut[2] * prob; // Loss
    final_wdl[3] += expertOut[3] * prob; // Mate
  }

  // 5. Convert to centipawns using cached converter
  float raw_eval = wdlConverter_.convert(final_wdl[0], final_wdl[1], final_wdl[2], final_wdl[3]);

  // 6. Material scaling
  int total_mat = 0;
  total_mat += board.pieces(PieceType::PAWN).count() * 100;
  total_mat += board.pieces(PieceType::KNIGHT).count() * 320;
  total_mat += board.pieces(PieceType::BISHOP).count() * 330;
  total_mat += board.pieces(PieceType::ROOK).count() * 500;
  total_mat += board.pieces(PieceType::QUEEN).count() * 900;
  
  if (!enableEvalNormalization_) {
    return raw_eval;
  }

  float multiplier = (static_cast<float>(evalScaleBase_) + static_cast<float>(total_mat) / static_cast<float>(evalScaleWeight_)) / 1024.0f;
  
  // Don't trade if ahead but very little material remains
  Color us = board.sideToMove();
  bool weAreWinning = (raw_eval > 0.0f);
  bool theyHaveNoPawns = board.pieces(PieceType::PAWN, ~us).empty();
  bool weHaveNoPawns = board.pieces(PieceType::PAWN, us).empty();
  
  if ((weAreWinning && theyHaveNoPawns) || (!weAreWinning && weHaveNoPawns) || (total_mat < 4000)) {
    multiplier -= 0.1f;
  }
  
  float final_ret = raw_eval * multiplier;
  
  // Clamp to avoid exceeding mate scores (MATE_SCORE is 31500, max WDL output is 30000)
  // If we scale a near-mate evaluation by 1.4x, it could exceed 31500 and corrupt search bounds/TT
  return std::clamp(final_ret, -30000.0f, 30000.0f);
}

WDLConverter::WDL EvalContextMoE::evaluateWDL(const Board &board) {
  // 1. Extract features and fill input buffers
  ChessInput features{};
  FeatureExtractor::fill_input(board, features);
  
  // Copy to aligned buffers (these are bound to the backbone inputs)
  std::memcpy(layersBuffer_.data(), features.layers, 32 * 64);
  std::memcpy(globalBuffer_.data(), features.global, 15 * sizeof(float));

  // 2. Compute routing weights client-side
  ExpertRouter::ExpertWeights weights;
  ExpertRouter::compute_weights(features, 0.0f, weights);

  // Get Top-2 Experts
  int top_idx[2];
  float top_probs[2];
  ExpertRouter::get_top_k(weights, 2, top_idx, top_probs);

  // Optimization: If primary expert is very confident, skip second
  int k_limit = 2;
  if (top_probs[0] > 0.90f) {
    k_limit = 1;
    top_probs[0] = 1.0f;
  }

  // 3. Run backbone using IO binding (zero allocations!)
  // SynchronizeInputs() handles copying from CPU buffer to device if using GPU
  backboneBinding_->SynchronizeInputs();
  backboneSession_->Run(Ort::RunOptions{nullptr}, *backboneBinding_);
  backboneBinding_->SynchronizeOutputs();
  
  // Backbone output is now in backboneOutputBuffer_, which is also bound
  // as the input to all expert bindings (zero-copy routing!)

  // 4. Run selected experts and accumulate WDL
  float final_wdl[4] = {0.0f};

  for (int k = 0; k < k_limit; ++k) {
    int expertIdx = top_idx[k];
    float prob = top_probs[k];

    // Run expert using IO binding (zero allocations, zero-copy input!)
    expertBindings_[expertIdx]->SynchronizeInputs();
    expertSessions_[expertIdx]->Run(Ort::RunOptions{nullptr}, *expertBindings_[expertIdx]);
    expertBindings_[expertIdx]->SynchronizeOutputs();
    
    // Expert output is in expertOutputBuffer_
    const float* expertOut = expertOutputBuffer_.data();

    // Weighted sum of WDL outputs
    final_wdl[0] += expertOut[0] * prob; // Win
    final_wdl[1] += expertOut[1] * prob; // Draw
    final_wdl[2] += expertOut[2] * prob; // Loss
    final_wdl[3] += expertOut[3] * prob; // Mate
  }

  // 5. Return raw WDL struct
  return {final_wdl[0], final_wdl[1], final_wdl[2], final_wdl[3]};
}

EvalContextMoE::ComponentTimings EvalContextMoE::benchmarkComponents(const Board &board) {
  using Clock = std::chrono::high_resolution_clock;
  using us = std::chrono::microseconds;
  ComponentTimings t;

  auto totalStart = Clock::now();

  // 1. Feature extraction + routing
  auto featStart = Clock::now();
  ChessInput features{};
  FeatureExtractor::fill_input(board, features);
  std::memcpy(layersBuffer_.data(), features.layers, 32 * 64);
  std::memcpy(globalBuffer_.data(), features.global, 15 * sizeof(float));

  ExpertRouter::ExpertWeights weights;
  ExpertRouter::compute_weights(features, 0.0f, weights);
  int top_idx[2];
  float top_probs[2];
  ExpertRouter::get_top_k(weights, 2, top_idx, top_probs);
  auto featEnd = Clock::now();
  t.featureExtractUs = std::chrono::duration_cast<us>(featEnd - featStart).count();
  t.topExpert0 = top_idx[0];
  t.topExpert1 = top_idx[1];
  t.topProb0 = top_probs[0];
  t.topProb1 = top_probs[1];

  // 2. Backbone
  auto bbStart = Clock::now();
  backboneBinding_->SynchronizeInputs();
  backboneSession_->Run(Ort::RunOptions{nullptr}, *backboneBinding_);
  backboneBinding_->SynchronizeOutputs();
  auto bbEnd = Clock::now();
  t.backboneUs = std::chrono::duration_cast<us>(bbEnd - bbStart).count();

  // 3. Run ALL 4 experts (for benchmarking purposes)
  double expertTimes[4] = {0};
  for (int e = 0; e < 4; ++e) {
    auto eStart = Clock::now();
    expertBindings_[e]->SynchronizeInputs();
    expertSessions_[e]->Run(Ort::RunOptions{nullptr}, *expertBindings_[e]);
    expertBindings_[e]->SynchronizeOutputs();
    auto eEnd = Clock::now();
    expertTimes[e] = std::chrono::duration_cast<us>(eEnd - eStart).count();
  }
  t.expert0Us = expertTimes[0];
  t.expert1Us = expertTimes[1];
  t.expert2Us = expertTimes[2];
  t.expert3Us = expertTimes[3];

  // 4. WDL conversion (run top-2 experts for realistic result)
  auto wdlStart = Clock::now();
  float final_wdl[4] = {0.0f};
  for (int k = 0; k < 2; ++k) {
    // Re-run the selected expert to get output (experts share output buffer)
    expertBindings_[top_idx[k]]->SynchronizeInputs();
    expertSessions_[top_idx[k]]->Run(Ort::RunOptions{nullptr}, *expertBindings_[top_idx[k]]);
    expertBindings_[top_idx[k]]->SynchronizeOutputs();
    const float* expertOut = expertOutputBuffer_.data();
    float prob = (top_probs[0] > 0.90f && k == 0) ? 1.0f : top_probs[k];
    if (top_probs[0] > 0.90f && k == 1) break;
    for (int i = 0; i < 4; ++i) final_wdl[i] += expertOut[i] * prob;
  }
  wdlConverter_.convert(final_wdl[0], final_wdl[1], final_wdl[2], final_wdl[3]);
  auto wdlEnd = Clock::now();
  t.wdlConvertUs = std::chrono::duration_cast<us>(wdlEnd - wdlStart).count();

  auto totalEnd = Clock::now();
  t.totalUs = std::chrono::duration_cast<us>(totalEnd - totalStart).count();

  return t;
}
