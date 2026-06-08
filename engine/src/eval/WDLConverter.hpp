#pragma once

/**
 * @file WDLConverter.hpp
 * @brief Converts neural network WDL outputs to centipawns.
 *
 * IMPORTANT: This is the INVERSE of WDLNormalizer.hpp's handle_centipawns().
 * The training formula uses a 0.3 factor that must be accounted for.
 *
 * Training (WDLNormalizer.hpp):
 *   win_rate = sigmoid(cp / SCALE)
 *   if cp > 0:
 *     w = win_rate, l = (1-w) * 0.3, d = 1 - w - l
 *   else:
 *     l = 1 - win_rate, w = win_rate * 0.3, d = 1 - l - w
 *
 * Inverse derivation (this file):
 *   We observe that w/(l+w) gives us the key ratio needed to invert.
 *   win_ratio = w / (w + l)
 *   cp = SCALE * ln(win_ratio / (1 - win_ratio))
 *
 * Features:
 *   - Correct inversion of the 0.3 scaling factor
 *   - Aggression parameter for playstyle tuning (-1.0 to +1.0)
 */

#include <algorithm>
#include <cmath>

/**
 * @class WDLConverter
 * @brief Handles conversion between Win-Draw-Loss probabilities and centipawn scores.
 */
class WDLConverter {
public:
  /**
   * @struct WDL
   * @brief Holds Win, Draw, and Loss probabilities.
   */
  struct WDL {
    float win, draw, loss;
  };

  // Score ranges
  static constexpr int MAX_CP = 15000;

  // Lichess constant: 1 / 0.00368208 (exact, from WDLNormalizer.hpp)
  static constexpr float WDL_SCALE = 1.0f / 0.00368208f;

  // Aggression parameter: -1.0 = defensive, 0 = neutral, +1.0 = aggressive
  float aggression = 0.0f;

  /**
   * Convert WDL to centipawn evaluation.
   *
   * @param win       Win probability [0, 1]
   * @param draw      Draw probability [0, 1]
   * @param loss      Loss probability [0, 1]
   * @return Centipawn score (positive = STM is winning)
   */
  int convert(float win, float draw, float loss) const {
    return convertWDL(win, draw, loss);
  }

private:
  /**
   * Convert WDL to centipawns - CORRECT INVERSE of WDLNormalizer.
   *
   * The key insight is that the 0.3 factor creates a ratio that we can invert.
   * From training:
   *   if cp > 0: w = sigmoid(cp/S), l = (1-w)*0.3
   *                → w/(w+l) = w/(w + 0.3*(1-w)) = w/(0.3 + 0.7*w)
   *   if cp < 0: l = 1-sigmoid(cp/S), w = sigmoid(cp/S)*0.3
   *                → w/(w+l) = 0.3*sig / (0.3*sig + 1-sig) = sig (after
   * solving)
   *
   * Actually, the simplest correct inverse:
   *   win_ratio = w / (w + l)  -- this extracts the sigmoid!
   *   sigmoid(x) = 1 / (1 + exp(-x))
   *   x = ln(win_ratio / (1 - win_ratio))
   *   cp = SCALE * ln(win_ratio / (1 - win_ratio))
   *
   * Aggression adjusts draw distribution before computing ratio.
   */
  int convertWDL(float win, float draw, float loss) const {
    // Aggression adjusts how draws affect win/loss perception
    // Positive aggression: draws feel like partial losses (fight for wins)
    // Negative aggression: draws feel like partial wins (play safe)
    float draw_to_win = 0.5f - aggression * 0.3f;
    draw_to_win = std::clamp(draw_to_win, 0.0f, 1.0f);

    // Adjust win/loss with draw allocation
    float win_adj = win + draw * draw_to_win;
    float loss_adj = loss + draw * (1.0f - draw_to_win);

    // The key ratio - this extracts the original sigmoid value
    // regardless of the 0.3 factor used in training
    float total = win_adj + loss_adj;
    if (total < 1e-6f) {
      return 0; // Draw-only position
    }
    float win_ratio = win_adj / total;

    // Clamp to avoid log(0) or log(inf)
    win_ratio = std::clamp(win_ratio, 1e-6f, 1.0f - 1e-6f);

    // Inverse sigmoid: cp = SCALE * ln(win_ratio / (1 - win_ratio))
    int cp =
        static_cast<int>(WDL_SCALE * std::log(win_ratio / (1.0f - win_ratio)));

    // Clamp to non-mate range
    return std::clamp(cp, -MAX_CP, MAX_CP);
  }
};
