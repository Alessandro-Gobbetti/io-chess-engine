/**
 * @file EvalContextMoECache.cpp
 * @brief Implementation of the thread-local MoE evaluator.
 *
 * Implements the incremental accumulator state updates and rapid board 
 * evaluation logic using the Factorized Mixture of Experts (MoE) network.
 * @ingroup engine
 */
#include "EvalContextMoECache.h"

#include "MoERouting.h"

#include <FactorizedFeatureExtractor.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

uint32_t EvalContextMoECache::read_u32(std::ifstream &in) {
  uint32_t v = 0;
  in.read(reinterpret_cast<char *>(&v), sizeof(v));
  if (!in.good())
    throw std::runtime_error("Failed reading u32");
  return v;
}

void EvalContextMoECache::read_floats_raw(std::ifstream &in, float *dst,
                                          size_t n) {
  in.read(reinterpret_cast<char *>(dst),
          static_cast<std::streamsize>(n * sizeof(float)));
  if (!in.good())
    throw std::runtime_error("Failed reading float block");
}

ExpertPoolMode EvalContextMoECache::pool_mode_from_code(int code) {
  switch (code) {
  case 0:
    return ExpertPoolMode::Flat;
  case 1:
    return ExpertPoolMode::Gap;
  case 2:
    return ExpertPoolMode::Pool2x2Avg;
  case 3:
    return ExpertPoolMode::Pool2x2Max;
  default:
    return ExpertPoolMode::Flat;
  }
}

void EvalContextMoECache::load_weights_into_target(
    const std::string &weightsPath, BenchConfig &cfg,
  SharedMoEWeights &weights) {
  std::ifstream in(weightsPath, std::ios::binary);
  if (!in.is_open())
    throw std::runtime_error("Cannot open weights file: " + weightsPath);

  const uint32_t magic = read_u32(in);
  const uint32_t version = read_u32(in);
  if (magic != kMagicWeights || version != kVersion)
    throw std::runtime_error("Invalid weights header");

  cfg.branchDim = static_cast<int>(read_u32(in));
  cfg.mixerOut = static_cast<int>(read_u32(in));
  cfg.nBypass = static_cast<int>(read_u32(in));
  cfg.nGlobals = static_cast<int>(read_u32(in));
  cfg.nExperts = static_cast<int>(read_u32(in));
  cfg.expertBottleneck = static_cast<int>(read_u32(in));
  cfg.expertHidden = static_cast<int>(read_u32(in));
  cfg.expertPoolMode = pool_mode_from_code(static_cast<int>(read_u32(in)));
  const int numBranches = static_cast<int>(read_u32(in));
  const int maxBranchPlanes = static_cast<int>(read_u32(in));

  if (numBranches != 12)
    throw std::runtime_error("Expected 12 branches");
  if (cfg.branchDim != NET_BRANCH_DIM || cfg.mixerOut != NET_MIXER_OUT ||
      cfg.nBypass != NET_BYPASS || cfg.nGlobals != NET_GLOBALS ||
      cfg.nExperts != NET_EXPERTS ||
      cfg.expertBottleneck != NET_EXPERT_BOTTLENECK ||
      cfg.expertHidden != NET_EXPERT_HIDDEN) {
    std::ostringstream oss;
    oss << "Weights dimensions do not match fixed native architecture: "
        << "bin=[bd=" << cfg.branchDim << ", nf=" << cfg.mixerOut
        << ", bypass=" << cfg.nBypass << ", globals=" << cfg.nGlobals
        << ", experts=" << cfg.nExperts << ", ebo="
        << cfg.expertBottleneck << ", eh=" << cfg.expertHidden << "] "
        << "expected=[bd=" << NET_BRANCH_DIM << ", nf=" << NET_MIXER_OUT
        << ", bypass=" << NET_BYPASS << ", globals=" << NET_GLOBALS
        << ", experts=" << NET_EXPERTS << ", ebo="
        << NET_EXPERT_BOTTLENECK << ", eh=" << NET_EXPERT_HIDDEN << "]";
    throw std::runtime_error(oss.str());
  }
  if (maxBranchPlanes != FactorizedInput::MAX_BRANCH_PLANES)
    throw std::runtime_error("weights maxBranchPlanes mismatch");

  std::array<int, 12> planesPerType{};
  for (int i = 0; i < 12; ++i)
    planesPerType[(size_t)i] = static_cast<int>(read_u32(in));

  for (int i = 0; i < 12; ++i) {
    if (planesPerType[(size_t)i] != active_channels_for_branch(i))
      throw std::runtime_error(
          "weights planes-per-branch mismatch with feature extractor");
  }

  cfg.nThreads = 1;
  weights.init_architecture(cfg.branchConvLayers);

  for (int b = 0; b < 12; ++b) {
    auto &br = weights.branches[(size_t)b];
    const int ic = active_channels_for_branch(b);

    read_floats(in, br.l0.w, (size_t)cfg.branchDim * ic * 9);
    read_floats(in, br.l0.b, (size_t)cfg.branchDim);
    read_floats(in, br.l1.w, (size_t)cfg.branchDim * 9);
    read_floats(in, br.l1.b, (size_t)cfg.branchDim);
    read_floats(in, br.l2.w, (size_t)cfg.branchDim * cfg.branchDim);
    read_floats(in, br.l2.b, (size_t)cfg.branchDim);
  }

  // Export stores mixer as one contiguous matrix [mixerOut][12*branchDim+nBypass].
  // Re-split into runtime layout expected by MoEDoubleAccumulator.
  std::vector<float> mixerW;
  const int mixerIn = 12 * cfg.branchDim + cfg.nBypass;
  read_floats(in, mixerW, (size_t)cfg.mixerOut * mixerIn);
  for (int oc = 0; oc < cfg.mixerOut; ++oc) {
    const float *row = &mixerW[(size_t)oc * mixerIn];
    for (int b = 0; b < 12; ++b) {
      for (int c = 0; c < cfg.branchDim; ++c) {
        weights.mixerWBr[((size_t)b * cfg.mixerOut + oc) * cfg.branchDim + c] =
            row[b * cfg.branchDim + c];
      }
    }
    for (int bp = 0; bp < cfg.nBypass; ++bp) {
      weights.mixerWBp[(size_t)bp * cfg.mixerOut + oc] =
          row[12 * cfg.branchDim + bp];
    }
  }
  read_floats(in, weights.mixerB, (size_t)cfg.mixerOut);
  read_floats(in, weights.globalW, (size_t)cfg.mixerOut * cfg.nGlobals);
  read_floats(in, weights.globalB, (size_t)cfg.mixerOut);
  read_floats(in, weights.gateW, (size_t)cfg.nExperts * cfg.nGlobals);
  read_floats(in, weights.gateB, (size_t)cfg.nExperts);

  const int hiddenIn =
      (cfg.expertPoolMode == ExpertPoolMode::Flat)
          ? (cfg.expertBottleneck * 64)
          : (cfg.expertPoolMode == ExpertPoolMode::Gap)
                ? cfg.expertBottleneck
                : (cfg.expertBottleneck * kPool2x2Regions);

  for (int e = 0; e < cfg.nExperts; ++e) {
    auto &ex = weights.experts[(size_t)e];
    read_floats(in, ex.wConv, (size_t)cfg.expertBottleneck * cfg.mixerOut);
    read_floats(in, ex.bConv, (size_t)cfg.expertBottleneck);

    std::vector<float> wHidden;
    read_floats(in, wHidden, (size_t)cfg.expertHidden * hiddenIn);

    if (cfg.expertPoolMode == ExpertPoolMode::Flat) {
      std::copy(wHidden.begin(), wHidden.end(), ex.wH.begin());
      transpose_copy(ex.wH, ex.wHT, cfg.expertHidden, hiddenIn);
    } else if (cfg.expertPoolMode == ExpertPoolMode::Gap) {
      std::copy(wHidden.begin(), wHidden.end(), ex.wHG.begin());
      transpose_copy(ex.wHG, ex.wHGT, cfg.expertHidden, hiddenIn);
    } else {
      std::copy(wHidden.begin(), wHidden.end(), ex.wH16.begin());
      transpose_copy(ex.wH16, ex.wH16T, cfg.expertHidden, hiddenIn);
    }

    read_floats(in, ex.bH, (size_t)cfg.expertHidden);
    read_floats(in, ex.wWdl, (size_t)3 * cfg.expertHidden);
    read_floats(in, ex.bWdl, (size_t)3);
  }
}

void EvalContextMoECache::load_weights_into_model(const std::string &weightsPath) {
  init_from_shared_model(loadSharedModel(weightsPath));
}

std::shared_ptr<const EvalContextMoECacheSharedModel>
EvalContextMoECache::loadSharedModel(const std::string &weightsPath) {
  auto shared = std::make_shared<EvalContextMoECacheSharedModel>();
  load_weights_into_target(weightsPath, shared->cfg, shared->weights);
  return shared;
}

void EvalContextMoECache::init_from_shared_model(
    std::shared_ptr<const EvalContextMoECacheSharedModel> shared) {
  if (!shared)
    throw std::runtime_error("Null shared native model handle");

  sharedModel_ = std::move(shared);
  cfg_ = sharedModel_->cfg;
  models_[0].init(&sharedModel_->weights, cfg_);
  models_[1].init(&sharedModel_->weights, cfg_);
  hasPrevByStm_[0] = false;
  hasPrevByStm_[1] = false;
  evalsSinceFullByStm_[0] = 0;
  evalsSinceFullByStm_[1] = 0;
}

EvalContextMoECache::EvalContextMoECache(const std::string &weightsPath) {
  load_weights_into_model(weightsPath);
  models_[0].initialized = false;
  models_[1].initialized = false;
}

EvalContextMoECache::EvalContextMoECache(
    std::shared_ptr<const EvalContextMoECacheSharedModel> sharedModel) {
  init_from_shared_model(std::move(sharedModel));
  models_[0].initialized = false;
  models_[1].initialized = false;
}

WDLConverter::WDL EvalContextMoECache::evaluateWDL(const Board &board,
                                                   int ply) {
  (void)ply;
  FactorizedInput &cur = scratchInput_;
  FactorizedFeatureExtractor::fill_input_rich(board, cur);

  const int stm_bucket = (board.sideToMove() == Color::WHITE) ? 1 : 0;
  MoEDoubleAccumulator &model = models_[(size_t)stm_bucket];
  FactorizedInput &prevInput = prevInputByStm_[(size_t)stm_bucket];
  bool &hasPrev = hasPrevByStm_[(size_t)stm_bucket];

  int active_e0 = 0;
  int active_e1 = 1;
  float active_w0 = 0.5f;
  float active_w1 = 0.5f;
  MoERouting::route_top2_base_experts(cur, active_e0, active_e1, active_w0,
                                      active_w1);
  const int activeExpertCount =
      MoERouting::collapse_top2_if_dominant(active_w0, active_w1);
  const bool useDominantExpertOnly = (activeExpertCount == 1);
  const int active_experts[2] = {active_e0, active_e1};

  bool didFullRebuild = false;
  const bool forcePeriodicRebuild =
      rebuildEveryNEvals_ > 0 &&
      evalsSinceFullByStm_[(size_t)stm_bucket] >=
          static_cast<uint32_t>(rebuildEveryNEvals_);

  if (!hasPrev || forcePeriodicRebuild) {
    model.full_rebuild_accumulators(cur, active_experts, activeExpertCount);
    hasPrev = true;
    didFullRebuild = true;
  } else {
    std::array<int, 12> dirty{};
    int dirtyCount = 0;
    for (int b = 0; b < 12; ++b) {
      if (branch_planes_changed(cur, prevInput, b))
        dirty[(size_t)dirtyCount++] = b;
    }

    if (dirtyCount > 6) {
      model.full_rebuild_accumulators(cur, active_experts, activeExpertCount);
      didFullRebuild = true;
    } else if (dirtyCount > 0) {
      model.update_incremental(cur, prevInput, dirty.data(), dirtyCount,
                               active_experts, activeExpertCount);
    }
  }

  float wdl[3] = {0.0f, 0.0f, 0.0f};
  if (useDominantExpertOnly) {
    model.run_active_expert(active_e0, wdl);
  } else {
    model.run_top2_experts(active_e0, active_e1, active_w0, active_w1, wdl);
  }

  prevInput = cur;

  if (didFullRebuild) {
    totalRebuilds_.fetch_add(1, std::memory_order_relaxed);
    evalsSinceFullByStm_[(size_t)stm_bucket] = 0;
  } else {
    ++evalsSinceFullByStm_[(size_t)stm_bucket];
  }

  return {wdl[0], wdl[1], wdl[2]};
}

float EvalContextMoECache::evaluate(const Board &board, int ply) {
  const WDLConverter::WDL wdl = evaluateWDL(board, ply);
  const float raw_eval =
      wdlConverter_.convert(wdl.win, wdl.draw, wdl.loss);

  int total_mat = 0;
  total_mat += board.pieces(PieceType::PAWN).count() * 100;
  total_mat += board.pieces(PieceType::KNIGHT).count() * 320;
  total_mat += board.pieces(PieceType::BISHOP).count() * 330;
  total_mat += board.pieces(PieceType::ROOK).count() * 500;
  total_mat += board.pieces(PieceType::QUEEN).count() * 900;

  if (!enableEvalNormalization_)
    return raw_eval;

  float multiplier = (static_cast<float>(evalScaleBase_) +
                      static_cast<float>(total_mat) /
                          static_cast<float>(evalScaleWeight_)) /
                     1024.0f;

  const Color us = board.sideToMove();
  const bool weAreWinning = (raw_eval > 0.0f);
  const bool theyHaveNoPawns = board.pieces(PieceType::PAWN, ~us).empty();
  const bool weHaveNoPawns = board.pieces(PieceType::PAWN, us).empty();

  if ((weAreWinning && theyHaveNoPawns) || (!weAreWinning && weHaveNoPawns) ||
      (total_mat < 4000)) {
    multiplier -= 0.1f;
  }

  const float scaled = raw_eval * multiplier;
  return std::clamp(scaled, -30000.0f, 30000.0f);
}
