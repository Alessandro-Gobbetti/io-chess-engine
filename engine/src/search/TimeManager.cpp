#include "TimeManager.h"
#include "../uci/UCI.h" // For SearchParams
#include <algorithm>
#include <cmath>

using chess::Color;

// =============================================================================
// Time Control Detection
// =============================================================================

TimeManager::TimeControl TimeManager::detectTimeControl(int baseTimeMs,
                                                        int incrementMs) const {
  // Correspondence: > 1 hour base with no/tiny increment
  if (baseTimeMs > 3600000 && incrementMs < 5000) {
    return TimeControl::CORRESPONDENCE;
  }

  // Consider effective time per move with increment
  // For a 40-move game: effectiveTime = base + 40 * increment
  int effectiveTime = baseTimeMs + 40 * incrementMs;

  // Bullet: < 2 minutes effective
  if (effectiveTime < 120000) {
    return TimeControl::BULLET;
  }
  // Blitz: < 5 minutes effective
  if (effectiveTime < 300000) {
    return TimeControl::BLITZ;
  }
  // Rapid: < 15 minutes effective
  if (effectiveTime < 900000) {
    return TimeControl::RAPID;
  }
  // Classical: >= 15 minutes
  return TimeControl::CLASSICAL;
}

// =============================================================================
// Multiplier Functions
// =============================================================================

float TimeManager::getPhaseMultiplier(float phase) const {
  // phase: 0.0 = opening, 1.0 = endgame
  // Smooth interpolation between phases

  if (phase < 0.15f) {
    // Opening: spend less time (rely on book/theory)
    return config_.openingMultiplier;
  } else if (phase < 0.55f) {
    // Middlegame: critical phase, think hard
    // Interpolate from opening to peak middlegame
    float t = (phase - 0.15f) / 0.40f; // 0 to 1 over middlegame
    float peak = config_.middlegameMultiplier;
    float start = config_.openingMultiplier;
    // Bell curve: peak at t=0.5
    float bellCurve = 1.0f - 4.0f * (t - 0.5f) * (t - 0.5f);
    return start + (peak - start) * bellCurve;
  } else {
    // Endgame: simpler positions
    float t = (phase - 0.55f) / 0.45f; // 0 to 1 over endgame
    return config_.middlegameMultiplier +
           (config_.endgameMultiplier - config_.middlegameMultiplier) * t;
  }
}

float TimeManager::getEvalMultiplier(float evalCp) const {
  float absEval = std::abs(evalCp);

  // Dead equal: most critical - tiny margins matter
  if (absEval < 30.0f) {
    return 1.5f;
  }

  // Slight edge either way: still important decisions
  if (absEval < 100.0f) {
    return 1.2f;
  }

  // Crushing or crushed: just play moves quickly
  // Position is decided, extra thinking won't help much
  if (absEval > 500.0f) {
    return 0.5f;
  }

  // Clearly winning: convert safely, don't overthink
  if (evalCp > 200.0f) {
    return config_.winningMultiplier; // 0.65
  }

  // Clearly losing: find swindle resources fast
  if (evalCp < -200.0f) {
    return config_.losingMultiplier; // 0.85
  }

  // Middle ground (100-200cp either way): normal thinking
  return 1.0f;
}

float TimeManager::getControlMultiplier(TimeControl control) const {
  switch (control) {
  case TimeControl::BULLET:
    return config_.bulletMultiplier;
  case TimeControl::BLITZ:
    return config_.blitzMultiplier;
  case TimeControl::RAPID:
    return config_.rapidMultiplier;
  case TimeControl::CLASSICAL:
    return config_.classicalMultiplier;
  case TimeControl::CORRESPONDENCE:
    return 1.0f; // Capped separately
  }
  return 1.0f;
}

int TimeManager::estimateMovesRemaining(int moveNumber, float phase,
                                        int movestogo) const {
  // If movestogo is specified, use it (add buffer)
  if (movestogo > 0) {
    return movestogo + 2;
  }

  // Estimate based on phase and move number
  // Average game is ~40 moves, but can vary

  // Opening (phase < 0.2): assume 35-40 moves left
  // Middlegame (0.2-0.6): assume 25-35 moves left
  // Endgame (> 0.6): assume 15-25 moves left

  int baseEstimate;
  if (phase < 0.2f) {
    baseEstimate = 38;
  } else if (phase < 0.6f) {
    baseEstimate = 30;
  } else {
    baseEstimate = 20;
  }

  // Adjust by actual move number (don't double-count)
  // If we're at move 30 and estimate says 30 left, that's total 60 - too high
  int remaining = std::max(15, baseEstimate - moveNumber / 3);

  return remaining;
}

// =============================================================================
// Main Calculation
// =============================================================================

TimeManager::TimeAllocation TimeManager::calculate(const SearchParams &params,
                                                   Color side, float phase,
                                                   float evalCp,
                                                   int moveNumber) const {
  TimeAllocation result{};

  // =============================================
  // Handle special cases
  // =============================================

  // Pondering or infinite: no time limit
  if (params.ponder || params.infinite) {
    result.softLimit = 0;
    result.hardLimit = 0;
    result.optimalTime = 0;
    result.panicTime = 0;
    result.control = TimeControl::CLASSICAL;
    return result;
  }

  // Fixed move time
  if (params.movetime > 0) {
    int64_t available =
        std::max(1, (params.movetime * 95) / 100 - config_.moveOverheadMs);
    result.softLimit = available;
    result.hardLimit = available;
    result.optimalTime = available;
    result.panicTime = available / 4;
    result.control = TimeControl::RAPID;
    return result;
  }

  // =============================================
  // Get our time and increment
  // =============================================
  int myTime = (side == Color::WHITE) ? params.wtime : params.btime;
  int myInc = (side == Color::WHITE) ? params.winc : params.binc;

  // No time control (depth/nodes only)
  if (myTime <= 0 && myInc <= 0) {
    result.softLimit = 0;
    result.hardLimit = 0;
    result.optimalTime = 0;
    result.panicTime = 0;
    result.control = TimeControl::CLASSICAL;
    return result;
  }

  // Only increment, no base time
  if (myTime <= 0) {
    myTime = myInc * 15; // Assume ~15 moves worth
  }

  // Detect time control
  result.control = detectTimeControl(myTime, myInc);

  // =============================================
  // Correspondence game handling
  // =============================================
  if (result.control == TimeControl::CORRESPONDENCE) {
    // Cap at reasonable time
    constexpr int64_t MAX_CORR_TIME = 60000; // 60 seconds
    result.softLimit = MAX_CORR_TIME / 2;
    result.hardLimit = MAX_CORR_TIME;
    result.optimalTime = MAX_CORR_TIME * 2 / 3;
    result.panicTime = MAX_CORR_TIME / 4;
    return result;
  }

  // =============================================
  // Calculate available time
  // =============================================
  int64_t availableTime =
      std::max(static_cast<int64_t>(config_.minTimeMs),
               static_cast<int64_t>(myTime - config_.moveOverheadMs));

  // =============================================
  // Estimate moves remaining
  // =============================================
  int movesRemaining =
      estimateMovesRemaining(moveNumber, phase, params.movestogo);

  // =============================================
  // Base time allocation
  // =============================================
  int64_t baseTime = availableTime / movesRemaining;

  // =============================================
  // Increment contribution
  // Smart increment handling:
  // - In early game, save more increment for later
  // - In late game, use more increment
  // - Never rely on increment too much in bullet
  // =============================================
  float incUsage = config_.incrementUsage;

  // Adjust based on time control
  if (result.control == TimeControl::BULLET) {
    incUsage *= 0.6f; // More conservative in bullet
  } else if (result.control == TimeControl::CLASSICAL) {
    incUsage *= 1.1f; // Can use more in classical
  }

  // Adjust based on moves played
  if (moveNumber < config_.minMovesForIncrement) {
    // Early game: save increment
    incUsage *= 0.5f;
  } else if (moveNumber > 40) {
    // Late game: use more increment
    incUsage = std::min(0.9f, incUsage * 1.3f);
  }

  int64_t incBonus = static_cast<int64_t>(myInc * incUsage);

  // =============================================
  // Apply multipliers
  // =============================================
  float phaseMultiplier = getPhaseMultiplier(phase);
  float evalMultiplier = getEvalMultiplier(evalCp);
  float controlMultiplier = getControlMultiplier(result.control);

  // Combined multiplier (cap to avoid extremes)
  float totalMultiplier = phaseMultiplier * evalMultiplier * controlMultiplier;
  totalMultiplier = std::clamp(totalMultiplier, 0.3f, 2.5f);

  // =============================================
  // Calculate optimal time
  // =============================================
  result.optimalTime =
      static_cast<int64_t>((baseTime + incBonus) * totalMultiplier);

  // =============================================
  // Calculate limits
  // =============================================

  // Max time: never use more than 15% of remaining time
  int64_t maxTime = availableTime * 15 / 100;

  // In bullet, be more conservative
  if (result.control == TimeControl::BULLET) {
    maxTime = availableTime * 10 / 100;
  }

  // CRITICAL: Zero-increment games need extra caution
  // No increment means no recovery - be very conservative
  if (myInc == 0) {
    if (result.control == TimeControl::BULLET) {
      // 0-increment bullet: max 5% of remaining time
      maxTime = std::min(maxTime, availableTime * 5 / 100);
    } else {
      // 0-increment blitz/rapid: max 7% of remaining time
      maxTime = std::min(maxTime, availableTime * 7 / 100);
    }
  }

  result.optimalTime = std::min(result.optimalTime, maxTime);

  // Soft limit: 70% of optimal (normal stopping point)
  result.softLimit = (result.optimalTime * 70) / 100;

  // Hard limit: 2x soft but capped at max
  result.hardLimit = std::min(result.softLimit * 2, maxTime);
  result.hardLimit = std::max(result.hardLimit, result.softLimit);

  // Panic threshold
  result.panicTime = config_.panicThresholdMs;

  // =============================================
  // Low time adjustments
  // =============================================

  // In bullet with no increment, start scaling earlier
  int lowTimeThreshold = (myInc == 0 && result.control == TimeControl::BULLET)
                             ? 20000
                             : 10000; // 20s vs 10s

  // Low time: scale down aggressively
  if (myTime < lowTimeThreshold) {
    float timeFactor =
        static_cast<float>(myTime) / static_cast<float>(lowTimeThreshold);
    result.softLimit =
        static_cast<int64_t>(result.softLimit * timeFactor * 0.5f);
    result.hardLimit =
        static_cast<int64_t>(result.hardLimit * timeFactor * 0.5f);
    result.optimalTime =
        static_cast<int64_t>(result.optimalTime * timeFactor * 0.5f);
  }

  // Panic mode: < 3 seconds
  if (myTime < config_.panicThresholdMs) {
    result.softLimit = std::min(result.softLimit, availableTime / 25);
    result.hardLimit = std::min(result.hardLimit, availableTime / 12);
  }

  // Ultra panic: < 1 second
  if (myTime < config_.ultraPanicMs) {
    result.softLimit = std::max(static_cast<int64_t>(15), availableTime / 50);
    result.hardLimit = std::max(static_cast<int64_t>(30), availableTime / 25);
  }

  // Ensure minimums
  result.softLimit =
      std::max(result.softLimit, static_cast<int64_t>(config_.minTimeMs));
  result.hardLimit =
      std::max(result.hardLimit, result.softLimit + config_.minTimeMs);
  result.optimalTime = std::max(result.optimalTime, result.softLimit);

  return result;
}

// =============================================================================
// Extension Factor (for instability)
// =============================================================================

float TimeManager::getExtensionFactor(int bestMoveChanges,
                                      int scoreDrops) const {
  float extension = 1.0f;

  // Extend for best move changes
  if (bestMoveChanges >= config_.instabilityThreshold) {
    // Logarithmic extension to avoid runaway
    float instability =
        static_cast<float>(bestMoveChanges - config_.instabilityThreshold + 1);
    extension += 0.15f * std::log2(instability + 1.0f);
  }

  // Extend for score drops (position getting worse)
  if (scoreDrops >= 2) {
    extension += 0.1f * std::min(scoreDrops, 5);
  }

  // Cap extension
  return std::min(extension, config_.maxInstabilityExtension);
}

// =============================================================================
// Should Stop
// =============================================================================

bool TimeManager::shouldStop(int64_t elapsed, int depth, int bestMoveChanges,
                             int scoreDrops,
                             const TimeAllocation &allocation) const {
  // No time limit
  if (allocation.softLimit <= 0) {
    return false;
  }

  // Always stop at hard limit
  if (elapsed >= allocation.hardLimit) {
    return true;
  }

  // Need minimum depth
  if (depth < 4) {
    return false;
  }

  // Get extension factor
  float extension = getExtensionFactor(bestMoveChanges, scoreDrops);
  int64_t adjustedSoftLimit =
      static_cast<int64_t>(allocation.softLimit * extension);

  // Stop if past adjusted soft limit
  if (elapsed >= adjustedSoftLimit) {
    return true;
  }

  // // At depth 6+, check if we have enough time for next iteration
  // // Assume next iteration takes 2-3x current time
  // if (depth >= 6 && elapsed > 0) {
  //   int64_t estimatedNextIteration = elapsed * 2;
  //   if (elapsed + estimatedNextIteration > allocation.hardLimit) {
  //     return true;
  //   }
  // }

  return false;
}
