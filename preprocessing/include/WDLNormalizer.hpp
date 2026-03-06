#pragma once
#include <algorithm>
#include <cmath>
#include <string>

/**
 * WDL (Win/Draw/Loss) + Mate Distance output structure.
 * Used as training labels for the MoE model.
 */
struct WDLOutput {
  float win;       // Probability of winning (0.0 - 1.0)
  float draw;      // Probability of draw (0.0 - 1.0)
  float loss;      // Probability of losing (0.0 - 1.0)
  float mate_dist; // Mate urgency: 0.0 = no mate, 1.0 = M1 (see below)
};

/**
 * WDLNormalizer - Converts eval strings to WDL probabilities + mate distance.
 *
 * Uses:
 *   - Sigmoid with SCALE=272 (Lichess constant) for centipawn → win probability
 *   - Exponential decay for mate distance: M1=0.607, M2=0.368, M10=0.007
 *   - Side-to-Move perspective (positive = we are winning)
 *
 * Mate Distance Scaling:
 *   Uses exponential decay: exp(-MATE_DECAY * N) where N = moves to mate.
 *   This gives more precision at low values:
 *     M1 = 0.607   (mate in 1 - very urgent)
 *     M2 = 0.368   (strong difference from M1)
 *     M3 = 0.223
 *     M5 = 0.082
 *     M10 = 0.007  (less urgent, already mostly solved)
 *   Compared to hyperbolic 1/(1+0.05*N):
 *     M1 = 0.952, M2 = 0.909 (too similar)
 */
class WDLNormalizer {
public:
  // Lichess constant: 1 / 0.00368208 = 271.58
  // Tuned on millions of real games
  static constexpr float SCALE = 1.0f / 0.00368208;

  // Mate decay: exponential provides better precision at low values
  // exp(-0.5*N): M1=0.607, M2=0.368, M5=0.082, M10=0.007
  static constexpr float MATE_DECAY = 0.5f;

  // Large CP value for mate scores (for router)
  static constexpr float MATE_CP = 15000.0f;

  /**
   * Parse eval string into WDL + mate distance (Side-to-Move perspective).
   *
   * CSV evals are from White's perspective. This function flips the result
   * when Black is to move, so the output is always from the current player's
   * view.
   *
   * @param eval_str         Raw eval string: "35", "-400", "#5", "#-3"
   * @param is_white_to_move True if White is the side to move
   * @return WDLOutput with probabilities summing to 1.0 (from STM perspective)
   */
  static WDLOutput convert(const std::string &eval_str, bool is_white_to_move) {
    WDLOutput result;

    // Check for mate scores
    if (eval_str.find('#') != std::string::npos) {
      result = handle_mate(eval_str);
    } else {
      // Handle centipawns
      float cp = 0.0f;
      try {
        cp = std::stof(eval_str);
      } catch (...) {
        // Fallback to equal position
        return {0.33f, 0.34f, 0.33f, 0.0f};
      }
      result = handle_centipawns(cp);
    }

    // Flip perspective for Black: swap Win <-> Loss
    if (!is_white_to_move) {
      std::swap(result.win, result.loss);
      // mate_dist stays the same (it's about urgency, not who wins)
    }

    return result;
  }

  /**
   * Extract raw centipawns for ExpertRouter (Killer/Survivor thresholds).
   * Returns value from Side-to-Move perspective: positive = STM is winning.
   *
   * @param eval_str         Raw eval string
   * @param is_white_to_move True if White is to move
   * @return Centipawn value (positive = side to move is winning)
   */
  static float to_centipawns(const std::string &eval_str,
                             bool is_white_to_move) {
    float cp = 0.0f;

    if (eval_str.find('#') != std::string::npos) {
      // Mate string → large CP value
      std::string s = eval_str;
      s.erase(std::remove(s.begin(), s.end(), '#'), s.end());
      int moves = std::stoi(s);
      int sign = (moves > 0) ? 1 : -1;
      // Cap at ±15000, decay slightly with move count
      cp = sign * (MATE_CP - std::abs(moves) * 10.0f);
    } else {
      try {
        cp = std::stof(eval_str);
      } catch (...) {
        return 0.0f;
      }
    }

    // Flip for Black (CSV is from White's perspective)
    if (!is_white_to_move) {
      cp = -cp;
    }

    return cp;
  }

private:
  /**
   * Convert centipawn score to WDL.
   * Uses logistic sigmoid: P(win) = 1 / (1 + exp(-cp/SCALE))
   */
  static WDLOutput handle_centipawns(float cp) {
    // Sigmoid for win probability
    float win_rate = 1.0f / (1.0f + std::exp(-cp / SCALE));

    float w, d, l;

    if (cp > 0) {
      // Winning position
      w = win_rate;
      l = (1.0f - w) * 0.3f; // Small loss chance
      d = 1.0f - w - l;      // Remainder is draw
    } else {
      // Losing position
      l = 1.0f - win_rate; // High loss probability
      w = win_rate * 0.3f; // Small win chance
      d = 1.0f - l - w;    // Remainder is draw
    }

    // Clamp and normalize
    w = std::max(0.0f, w);
    d = std::max(0.0f, d);
    l = std::max(0.0f, l);
    float sum = w + d + l;

    return {w / sum, d / sum, l / sum, 0.0f}; // No mate in CP positions
  }

  /**
   * Convert mate score to WDL.
   * "#5" = we mate in 5, "#-3" = they mate us in 3.
   *
   * Uses shifted exponential decay: exp(-MATE_DECAY * (N-1))
   * This maximizes the useful range since M0 doesn't exist:
   *   M1 = 1.000 (mate in 1 - maximum urgency)
   *   M2 = 0.607
   *   M3 = 0.368
   *   M5 = 0.135
   *   M10 = 0.011
   */
  static WDLOutput handle_mate(std::string s) {
    s.erase(std::remove(s.begin(), s.end(), '#'), s.end());
    int moves = std::stoi(s);

    // Shifted exponential decay: N-1 so that M1 = 1.0
    // exp(-0.5 * 0) = 1.0, exp(-0.5 * 1) = 0.607, exp(-0.5 * 9) = 0.011
    float n = static_cast<float>(std::abs(moves));
    float dist = std::exp(-MATE_DECAY * (n - 1.0f));

    if (moves > 0) {
      // We mate them: Win=1.0, with mate urgency
      return {1.0f, 0.0f, 0.0f, dist};
    } else {
      // They mate us: Loss=1.0, mate distance = how threatened we are
      return {0.0f, 0.0f, 1.0f, dist};
    }
  }
};
