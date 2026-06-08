#pragma once

/**
 * @file TimeManager.h
 * @brief Sophisticated time allocation and management for the chess engine.
 *
 * Features:
 *   - Time control detection (bullet/blitz/rapid/classical)
 *   - Game phase awareness (opening/middlegame/endgame)
 *   - Evaluation-based thinking (longer in critical positions)
 *   - Increment handling with reserve management
 *   - Instability extension (when best move keeps changing)
 *   - Panic mode for low time situations
 */

#include "chess.hpp" // For chess::Color
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

struct SearchParams;

/**
 * @class TimeManager
 * @brief Handles adaptive time allocation during search.
 */
class TimeManager {
public:
  /**
   * @enum TimeControl
   * @brief Categorization of time controls based on base time.
   */
  enum class TimeControl {
    BULLET,        ///< < 2 minutes base
    BLITZ,         ///< 2-5 minutes base
    RAPID,         ///< 5-15 minutes base
    CLASSICAL,     ///< > 15 minutes base
    CORRESPONDENCE ///< > 1 hour or untimed
  };

  /**
   * @struct TimeAllocation
   * @brief Recommended time limits for the current move.
   */
  struct TimeAllocation {
    int64_t softLimit;   ///< Normal stopping point (ms).
    int64_t hardLimit;   ///< Absolute maximum (ms).
    int64_t optimalTime; ///< Ideal time to use (ms).
    int64_t panicTime;   ///< Below this, enter panic mode.
    TimeControl control; ///< Detected time control type.
  };

  /**
   * @struct Config
   * @brief Tunable configuration for time management heuristics.
   */
  struct Config {
    // Phase multipliers (0.0 = opening, 1.0 = endgame)
    float openingMultiplier = 0.75f;    ///< Quick in opening (book/theory).
    float middlegameMultiplier = 1.25f; ///< Think hard in complex positions.
    float endgameMultiplier = 0.70f;    ///< Simpler, fewer variations.

    // Time control multipliers
    float bulletMultiplier = 0.55f;     ///< Very aggressive time saving.
    float blitzMultiplier = 0.75f;      ///< Balanced.
    float rapidMultiplier = 1.0f;       ///< Normal.
    float classicalMultiplier = 1.15f;  ///< Can think deeper.

    // Eval-based multipliers
    float criticalMultiplier = 1.4f;    ///< Unclear position (±50cp).
    float edgeMultiplier = 1.0f;        ///< Slight edge (50-150cp).
    float winningMultiplier = 0.65f;    ///< Clear advantage (>200cp).
    float losingMultiplier = 0.85f;     ///< Swindle mode (<-150cp).

    // Instability extension
    float maxInstabilityExtension = 2.0f; ///< Max extension factor.
    int instabilityThreshold = 3;         ///< Changes before extending.

    // Increment handling
    float incrementUsage = 0.7f;   ///< Use 70% of increment per move.
    float incrementReserve = 0.3f; ///< Keep 30% as buffer.
    int minMovesForIncrement = 5;  ///< Don't rely on inc for first moves.

    // Safety margins
    int moveOverheadMs = 50;     ///< Network latency buffer.
    int minTimeMs = 10;          ///< Absolute minimum thinking time.
    int panicThresholdMs = 3000; ///< Enter panic below this.
    int ultraPanicMs = 1000;     ///< Emergency mode.
  };

  TimeManager() = default;
  explicit TimeManager(const Config &cfg) : config_(cfg) {}

  /**
   * @brief Calculates time allocation for a position.
   *
   * @param params     Search parameters (time, inc, movestogo, etc.).
   * @param side       Side to move.
   * @param phase      Game phase from FeatureExtractor (0.0=opening, 1.0=endgame).
   * @param evalCp     Current evaluation in centipawns (from previous search).
   * @param moveNumber Full move number (1-based).
   * @return TimeAllocation with soft/hard limits.
   */
  TimeAllocation calculate(const SearchParams &params, chess::Color side,
                           float phase, float evalCp, int moveNumber) const;

  /**
   * @brief Checks if search should stop based on elapsed time and stability.
   *
   * @param elapsed          Time spent so far (ms).
   * @param depth            Current search depth.
   * @param bestMoveChanges  How many times best move changed.
   * @param scoreDrops       How many times score dropped significantly.
   * @param allocation       Previously calculated allocation limits.
   * @return True if the search should abort.
   */
  bool shouldStop(int64_t elapsed, int depth, int bestMoveChanges,
                  int scoreDrops, const TimeAllocation &allocation) const;

  /**
   * @brief Gets the extension factor based on search instability.
   * 
   * Called during iterative deepening to adjust time limits dynamically.
   * 
   * @param bestMoveChanges Count of root best move changes.
   * @param scoreDrops Count of significant score drops.
   * @return Multiplier to extend the soft limit.
   */
  float getExtensionFactor(int bestMoveChanges, int scoreDrops) const;

  const Config &config() const { return config_; }
  void setConfig(const Config &cfg) { config_ = cfg; }

private:
  Config config_;

  TimeControl detectTimeControl(int baseTimeMs, int incrementMs) const;
  float getPhaseMultiplier(float phase) const;
  float getEvalMultiplier(float evalCp) const;
  float getControlMultiplier(TimeControl control) const;
  int estimateMovesRemaining(int moveNumber, float phase, int movestogo) const;
};
