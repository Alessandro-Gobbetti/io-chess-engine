#pragma once

/**
 * @file IEvaluator.h
 * @brief Abstract interface for board evaluation algorithms.
 *
 * Defines the common API that all evaluators (both neural-network-based and
 * simple handcrafted ones) must implement to score board positions.
 */

#include "FeatureExtractor.hpp"
#include "WDLConverter.hpp"
#include "../Types.h"

#include <cstdint>

/**
 * @class IEvaluator
 * @brief Abstract interface for evaluators.
 *
 * Implementations of this interface are responsible for providing static
 * evaluations of chess positions, translating them into centipawn scores
 * or Win/Draw/Loss probabilities.
 */
class IEvaluator {
public:
  virtual ~IEvaluator() = default;

  /**
   * @brief Evaluates the board from the perspective of the side to move.
   * 
   * @param board The current chess board state.
   * @param ply The current depth from the root of the search (used for scaling).
   * @return Evaluation score in centipawns (e.g., 150 = +1.5 pawns advantage).
   */
  virtual float evaluate(const Board &board, int ply = 0) = 0;

  /**
   * @brief Evaluates the board and returns Win/Draw/Loss probabilities.
   * 
   * @param board The current chess board state.
   * @param ply The current depth from the root of the search.
   * @return WDL probability distribution.
   */
  virtual WDLConverter::WDL evaluateWDL(const Board &board, int ply = 0) {
    (void)board;
    (void)ply;
    // Default implementation for evaluators that don't support WDL natively (e.g., SimpleEval)
    return {0.0f, 0.0f, 0.0f};
  }

  /**
   * @brief Evaluates the position using pre-computed features.
   * 
   * This is an optional optimization path for evaluators that utilize incremental
   * feature extraction (like MoE neural networks).
   * 
   * @param input The pre-extracted neural network features.
   * @return Evaluation score in centipawns.
   */
  virtual float evaluate(const ChessInput &input) {
    (void)input;
    return 0.0f;
  }
  
  /**
   * @brief Sets the contempt or aggression factor for the evaluator.
   * 
   * @param aggression The aggression factor (positive encourages risk, negative encourages draws).
   */
  virtual void setAggression(float aggression) { (void)aggression; }

  /**
   * @brief Sets the scaling parameters for the evaluation score.
   * 
   * @param base The base scaling divisor.
   * @param weight The material weight scaling factor.
   */
  virtual void setEvalScale(int base, int weight) { (void)base; (void)weight; }

  /**
   * @brief Enables or disables dynamic evaluation normalization.
   * 
   * @param enable True to enable normalization, false to use raw scores.
   */
  virtual void setEvalNormalization(bool enable) { (void)enable; }

  /**
   * @brief Sets the interval for forcing full feature rebuilds (to correct accumulation errors).
   * 
   * @param interval The number of incremental updates before forcing a rebuild.
   */
  virtual void setIncrementalRebuildInterval(int interval) {
    (void)interval;
  }

  /**
   * @brief Retrieves the number of full feature rebuilds performed (for profiling).
   * 
   * @return The number of full rebuilds.
   */
  virtual uint64_t getFullRebuilds() const { return 0; }
};
