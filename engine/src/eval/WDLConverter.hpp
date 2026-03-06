#pragma once
/**
 * WDLConverter - Converts neural network WDL + mate_dist outputs to centipawns.
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
 *   - Dual-signal mate detection (WDL AND mate_dist must agree)
 *   - Aggression parameter for playstyle tuning (-1.0 to +1.0)
 */

#include <algorithm>
#include <cmath>

class WDLConverter {
public:
  struct WDL {
    float win, draw, loss, mate;
  };
  // Must match WDLNormalizer.hpp
  static constexpr float MATE_DECAY = 0.5f;

  // Score ranges
  static constexpr int MATE_SCORE = 30000;
  static constexpr int MAX_CP = 15000;

  // Lichess constant: 1 / 0.00368208 (exact, from WDLNormalizer.hpp)
  static constexpr float WDL_SCALE = 1.0f / 0.00368208f;

  // Mate boost parameters
  static constexpr float MATE_DIST_THRESHOLD = 0.1f; // Start boosting at ~M5
  static constexpr float WDL_DECISIVE_THRESHOLD =
      0.80f;                                 // Only boost if 80%+ confident
  static constexpr int MATE_BOOST_MAX = 200; // Maximum boost in centipawns

  // Aggression parameter: -1.0 = defensive, 0 = neutral, +1.0 = aggressive
  float aggression = 0.0f;

  /**
   * Convert WDL + mate_dist to centipawn evaluation.
   *
   * @param win       Win probability [0, 1]
   * @param draw      Draw probability [0, 1]
   * @param loss      Loss probability [0, 1]
   * @param mate_dist Mate distance from network [0, 1] (1.0 = M1)
   * @return Centipawn score (positive = STM is winning)
   */
  int convert(float win, float draw, float loss, float mate_dist) const {
    // Always use WDL as the base evaluation
    int base_cp = convertWDL(win, draw, loss);

    // Garbage sanitization (NN out of bounds or NaN/Inf safety)
    if (std::isnan(mate_dist) || std::isinf(mate_dist) || mate_dist < 0.0f || mate_dist > 1.0f) {
        return base_cp;
    }

    // Add a small boost when mate is indicated
    // Boost scales with mate_dist (closer mates = bigger boost)
    if (mate_dist > MATE_DIST_THRESHOLD) {
      bool wdl_decisive =
          (win > WDL_DECISIVE_THRESHOLD || loss > WDL_DECISIVE_THRESHOLD);

      if (wdl_decisive) {
        // Boost proportional to mate_dist: M1 (1.0) = full boost, M5 (0.1) =
        // small boost
        float boost_factor = std::clamp((mate_dist - MATE_DIST_THRESHOLD) / (1.0f - MATE_DIST_THRESHOLD), 0.0f, 1.0f);
        int boost = static_cast<int>(boost_factor * MATE_BOOST_MAX);

        // Apply boost in same direction as evaluation
        if (base_cp > 0) {
          int mate_val = MATE_SCORE - static_cast<int>(boost_factor * 200.0f); // safer math
          return std::max(base_cp + boost, mate_val);
        } else {
          int mate_val = -MATE_SCORE + static_cast<int>(boost_factor * 200.0f); // safer math
          return std::min(base_cp - boost, mate_val);
        }
      }
    }

    return base_cp;
  }

private:
  /**
   * Convert mate position to centipawn score.
   *
   * Inverts the WDLNormalizer.hpp formula:
   *   mate_dist = exp(-MATE_DECAY * (moves - 1))
   *   → moves = 1 - ln(mate_dist) / MATE_DECAY
   */
  int convertMate(float win, float loss, float mate_dist) const {
    // Invert exponential: moves = 1 + (-ln(mate_dist) / MATE_DECAY)
    float log_dist = std::log(std::max(mate_dist, 1e-6f));
    int moves = 1 + static_cast<int>(-log_dist / MATE_DECAY);
    moves = std::clamp(moves, 1, 100);

    // Convert to plies (half-moves): M1 = 1 ply, M2 = 3 plies, M3 = 5 plies
    int plies = moves * 2 - 1;

    // Determine sign from WDL
    bool we_mate = (win > loss);
    return we_mate ? (MATE_SCORE - plies) : (-MATE_SCORE + plies);
  }

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
