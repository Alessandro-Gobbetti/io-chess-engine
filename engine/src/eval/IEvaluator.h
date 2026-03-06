#pragma once
// IEvaluator.h - Interface for different evaluation methods

#include "FeatureExtractor.hpp"
#include "WDLConverter.hpp"
#include "../Types.h"

// Abstract interface for evaluators
// Both NN-based and simple evaluators implement this
class IEvaluator {
public:
  virtual ~IEvaluator() = default;

  // Evaluate board from side-to-move perspective
  // Returns evaluation in centipawns (e.g., 150 = +1.5 pawns advantage)
  virtual float evaluate(const Board &board) = 0;

  // Evaluate and return WDL + Mate probabilities
  virtual WDLConverter::WDL evaluateWDL(const Board &board) {
    // Default implementation for evaluators that don't support WDL (e.g. Simple)
    return {0.0f, 0.0f, 0.0f, 0.0f};
  }

  // Evaluate using pre-computed features (optional optimization)
  // Default implementation falls back to extracting features from board (if
  // possible) or throws
  virtual float evaluate(const ChessInput &input) {
    // Default: Evaluator might not support this, or we just ignore input if not
    // needed But ideally we want pure abstract or helpful default. Since
    // specific evaluators (NN) NEED input, and SimpleEval USES input... Let's
    // make it abstract? No, SimpleEvalContext might want different signature?
    // SimpleEvalContext currently has evaluate(Board, ChessInput).
    // Let's force implementations to handle it.
    return 0.0f;
  }
  // Optional parameters
  virtual void setAggression(float aggression) { (void)aggression; }
  virtual void setEvalScale(int base, int weight) { (void)base; (void)weight; }
  virtual void setEvalNormalization(bool enable) { (void)enable; }
};
