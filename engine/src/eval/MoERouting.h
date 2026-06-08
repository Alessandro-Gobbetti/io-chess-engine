#pragma once

/**
 * @file MoERouting.h
 * @brief Mixture of Experts (MoE) routing logic.
 *
 * Provides utility functions to select the top active experts for
 * the neural network based on the input features.
 */

#include <ExpertRouter.hpp>

#include <algorithm>

namespace MoERouting {

/**
 * @brief Collapses the top 2 experts into a single expert if the top one is highly dominant.
 * 
 * @param top1Weight Weight of the highest ranked expert (modified in place).
 * @param top2Weight Weight of the second highest ranked expert (modified in place).
 * @param dominantExpertCutoff Threshold weight above which the second expert is dropped.
 * @return Number of active experts (1 or 2).
 */
inline int collapse_top2_if_dominant(
    float &top1Weight, float &top2Weight,
    float dominantExpertCutoff = 0.8f) {
  if (top1Weight > dominantExpertCutoff) {
    // Renormalize to 1.0 when we intentionally skip the second expert.
    top1Weight = 1.0f;
    top2Weight = 0.0f;
    return 1;
  }
  return 2;
}

/**
 * @brief Identifies the top 2 base experts for a given factorized input.
 * 
 * @param inp The factorized input features.
 * @param e0 Output index of the primary expert.
 * @param e1 Output index of the secondary expert.
 * @param w0 Output normalized weight for the primary expert.
 * @param w1 Output normalized weight for the secondary expert.
 */
inline void route_top2_base_experts(const FactorizedInput &inp, int &e0,
                                    int &e1, float &w0, float &w1) {
  ExpertRouter::ExpertWeights weights{};
  ExpertRouter::compute_weights(inp, 0.0f, weights);

  e0 = 0;
  e1 = 1;
  float s0 = weights.weights[0];
  float s1 = weights.weights[1];
  if (s1 > s0) {
    std::swap(e0, e1);
    std::swap(s0, s1);
  }

  for (int e = 2; e < ExpertRouter::NUM_BASE; ++e) {
    const float s = weights.weights[e];
    if (s > s0) {
      s1 = s0;
      e1 = e0;
      s0 = s;
      e0 = e;
    } else if (s > s1) {
      s1 = s;
      e1 = e;
    }
  }

  const float z = s0 + s1;
  if (z > 1e-20f) {
    w0 = s0 / z;
    w1 = s1 / z;
  } else {
    w0 = 0.5f;
    w1 = 0.5f;
  }
}

} // namespace MoERouting
