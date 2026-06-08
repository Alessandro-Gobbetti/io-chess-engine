/**
 * @file ExpertRouter.hpp
 * @brief Computes expert weights for the Residual MoE architecture based on position features.
 */
#pragma once
#include "FactorizedFeatureExtractor.hpp"
#include "FeatureExtractor.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

/**
 * ExpertRouter - Computes expert weights for Residual MoE architecture
 *
 * RESIDUAL ARCHITECTURE:
 *   Base Experts (0-3): Softmax group, sum to 1.0 - provides foundation
 *   Aux Experts (4-5):  Independent gates (0-1) - provides residual bonus
 *
 * 6 Experts:
 *   BASE GROUP (Softmax):
 *     0: TACTICAL    - High CCT middlegame positions
 *     1: STRATEGIC   - Low CCT middlegame positions
 *     2: MAJOR_END   - Endgame with rooks/queens
 *     3: MINOR_END   - Endgame without major pieces
 *   AUX GROUP (Independent gates):
 *     4: SURVIVOR    - Resistance bonus when losing
 *     5: KILLER      - Progress bonus when winning
 */
class ExpertRouter {
public:
  static constexpr int NUM_EXPERTS = 6;
  static constexpr int NUM_BASE = 4; // Base experts (softmax group)
  static constexpr int NUM_AUX = 2;  // Auxiliary experts (independent gates)

  // Expert indices
  enum Expert : int {
    // Base Group (indices 0-3, softmax)
    TACTICAL = 0,
    STRATEGIC = 1,
    MAJOR_END = 2,
    MINOR_END = 3,
    // Aux Group (indices 4-5, independent)
    SURVIVOR = 4,
    KILLER = 5
  };

  // Output structure for routing (Residual MoE)
  // Binary format: 6 floats = [base0, base1, base2, base3, survivor, killer]
  // - base[0-3]: sum to 1.0 (softmax normalized)
  // - survivor/killer: independent 0.0 to 1.0 (gate activations)
  struct alignas(32) ExpertWeights {
    float weights[NUM_EXPERTS];    // Final weights for output
    float raw_scores[NUM_EXPERTS]; // Pre-normalization scores (debugging)
  };

  // =========================================================================
  // Thresholds (tunable)
  // =========================================================================
  static constexpr float PHASE_ENDGAME_THRESHOLD = 0.65f;

  // CCT thresholds
  static constexpr float CCT_TACTICAL_THRESHOLD = 14.0f;
  static constexpr float CCT_STRATEGIC_THRESHOLD = 6.0f;

  // Aux gate thresholds (in CENTIPAWNS)
  static constexpr float KILLER_CP_THRESHOLD =
      150.0f; // Activate when winning 1.5+ pawns
  static constexpr float SURVIVOR_CP_THRESHOLD =
      150.0f; // Activate when losing 1.5+ pawns
  static constexpr float AUX_RAMP_RANGE = 500.0f; // CP range for 0->1 ramp

  // CCT weights
  static constexpr float CCT_WEIGHT_CHECK = 10.0f;
  static constexpr float CCT_WEIGHT_MAJOR_THREAT = 3.0f;

  // Piece values for capture scoring
  static constexpr float PIECE_VALUE_PAWN = 1.0f;
  static constexpr float PIECE_VALUE_KNIGHT = 3.0f;
  static constexpr float PIECE_VALUE_BISHOP = 3.0f;
  static constexpr float PIECE_VALUE_ROOK = 5.0f;
  static constexpr float PIECE_VALUE_QUEEN = 9.0f;

  // Base expert score magnitudes (continuous scoring)
  // SCORE_MAX: maximum score for dominant expert
  // SCORE_MIN: minimum score to prevent softmax degeneration
  static constexpr float SCORE_MAX = 5.0f;
  static constexpr float SCORE_MIN = 0.5f;

  // =========================================================================
  // Main API
  // =========================================================================

  /**
   * Compute expert weights using Residual MoE architecture.
   *
   * Output format (6 floats):
   *   [0-3]: Base experts (TACTICAL, STRATEGIC, MAJOR_END, MINOR_END) -
   * softmax, sum=1 [4]:   SURVIVOR gate - independent 0-1 (active when losing)
   *   [5]:   KILLER gate   - independent 0-1 (active when winning)
   *
   * @param input   The ChessInput features
   * @param eval_cp Evaluation in centipawns (positive = side to move winning)
   * @param out     Output weights structure
   */
  static void compute_weights(const ChessInput &input, float eval_cp,
                              ExpertWeights &out) {
    // Compute derived features
    float phase = compute_phase(input); // 0.0 = opening, 1.0 = endgame
    float cct = compute_cct(input);     // Higher = more tactical
    float major_ratio =
        compute_major_ratio(input); // 0.0 = no majors, 1.0 = all majors

    // ========== GROUP A: Base Experts (Continuous Scores) ==========
    // Each expert gets a continuous score based on position features.
    // Softmax will produce natural weights that reflect the position's
    // character.

    float raw_base[NUM_BASE] = {0.0f};

    // --- Phase-based split: Middlegame vs Endgame ---
    // Smooth transition around PHASE_ENDGAME_THRESHOLD (0.65)
    // sigmoid centered at threshold, steepness controls blend zone
    float endgame_factor = sigmoid((phase - PHASE_ENDGAME_THRESHOLD) * 8.0f);
    float middlegame_factor = 1.0f - endgame_factor;

    // --- Middlegame: Tactical vs Strategic (CCT-based) ---
    // Smooth interpolation across entire CCT range
    float tactical_strength =
        sigmoid((cct - CCT_STRATEGIC_THRESHOLD) /
                    (CCT_TACTICAL_THRESHOLD - CCT_STRATEGIC_THRESHOLD) * 4.0f -
                2.0f);

    raw_base[TACTICAL] = middlegame_factor * tactical_strength * SCORE_MAX;
    raw_base[STRATEGIC] =
        middlegame_factor * (1.0f - tactical_strength) * SCORE_MAX;

    // --- Endgame: Major vs Minor (piece type based) ---
    // major_ratio: 0.0 = pure minor endgame, 1.0 = pure major endgame
    // Blend smoothly based on actual piece composition
    raw_base[MAJOR_END] = endgame_factor * major_ratio * SCORE_MAX;
    raw_base[MINOR_END] = endgame_factor * (1.0f - major_ratio) * SCORE_MAX;

    // --- Ensure minimum score for active experts ---
    // This prevents any expert from getting exactly 0 (softmax issues)
    for (int i = 0; i < NUM_BASE; ++i) {
      raw_base[i] = std::max(raw_base[i], SCORE_MIN);
    }

    // Softmax normalize base experts (sum to 1.0)
    softmax(raw_base, out.weights, NUM_BASE);

    // Copy raw scores for debugging / dataset selection
    for (int i = 0; i < NUM_BASE; ++i) {
      out.raw_scores[i] = raw_base[i];
    }

    // ========== GROUP B: Aux Experts (Independent Gates) ==========

    // SURVIVOR (Index 4): Linear ramp when losing
    if (eval_cp < -SURVIVOR_CP_THRESHOLD) {
      float deficit = std::abs(eval_cp) - SURVIVOR_CP_THRESHOLD;
      out.weights[SURVIVOR] = std::clamp(deficit / AUX_RAMP_RANGE, 0.0f, 1.0f);
    } else {
      out.weights[SURVIVOR] = 0.0f;
    }
    out.raw_scores[SURVIVOR] = out.weights[SURVIVOR]; // Gate value = raw score

    // KILLER (Index 5): Linear ramp when winning
    if (eval_cp > KILLER_CP_THRESHOLD) {
      float bonus = eval_cp - KILLER_CP_THRESHOLD;
      out.weights[KILLER] = std::clamp(bonus / AUX_RAMP_RANGE, 0.0f, 1.0f);
    } else {
      out.weights[KILLER] = 0.0f;
    }
    out.raw_scores[KILLER] = out.weights[KILLER]; // Gate value = raw score
  }

  /**
   * Compute expert weights directly from factorized features.
   * This avoids extracting ChessInput when factorized preprocessing is selected.
   */
  static void compute_weights(const FactorizedInput &input, float eval_cp,
                              ExpertWeights &out) {
    float phase = compute_phase(input);
    float cct = compute_cct(input);
    float major_ratio = compute_major_ratio(input);

    float raw_base[NUM_BASE] = {0.0f};

    float endgame_factor = sigmoid((phase - PHASE_ENDGAME_THRESHOLD) * 8.0f);
    float middlegame_factor = 1.0f - endgame_factor;

    float tactical_strength =
        sigmoid((cct - CCT_STRATEGIC_THRESHOLD) /
                    (CCT_TACTICAL_THRESHOLD - CCT_STRATEGIC_THRESHOLD) *
                    4.0f -
                2.0f);

    raw_base[TACTICAL] = middlegame_factor * tactical_strength * SCORE_MAX;
    raw_base[STRATEGIC] =
        middlegame_factor * (1.0f - tactical_strength) * SCORE_MAX;
    raw_base[MAJOR_END] = endgame_factor * major_ratio * SCORE_MAX;
    raw_base[MINOR_END] = endgame_factor * (1.0f - major_ratio) * SCORE_MAX;

    for (int i = 0; i < NUM_BASE; ++i) {
      raw_base[i] = std::max(raw_base[i], SCORE_MIN);
    }

    softmax(raw_base, out.weights, NUM_BASE);

    for (int i = 0; i < NUM_BASE; ++i) {
      out.raw_scores[i] = raw_base[i];
    }

    if (eval_cp < -SURVIVOR_CP_THRESHOLD) {
      float deficit = std::abs(eval_cp) - SURVIVOR_CP_THRESHOLD;
      out.weights[SURVIVOR] = std::clamp(deficit / AUX_RAMP_RANGE, 0.0f, 1.0f);
    } else {
      out.weights[SURVIVOR] = 0.0f;
    }
    out.raw_scores[SURVIVOR] = out.weights[SURVIVOR];

    if (eval_cp > KILLER_CP_THRESHOLD) {
      float bonus = eval_cp - KILLER_CP_THRESHOLD;
      out.weights[KILLER] = std::clamp(bonus / AUX_RAMP_RANGE, 0.0f, 1.0f);
    } else {
      out.weights[KILLER] = 0.0f;
    }
    out.raw_scores[KILLER] = out.weights[KILLER];
  }

  /**
   * Get top-k expert indices and their weights.
   *
   * @param weights  The computed expert weights
   * @param k        Number of experts to select (typically 2)
   * @param indices  Output array of expert indices (sorted by weight,
   * descending)
   * @param probs    Output array of normalized probabilities for selected
   * experts
   */
  static void get_top_k(const ExpertWeights &weights, int k, int *indices,
                        float *probs) {
    // Create index array
    int idx[NUM_EXPERTS] = {0, 1, 2, 3, 4, 5};

    // Partial sort to get top-k
    std::partial_sort(idx, idx + k, idx + NUM_EXPERTS,
                      [&weights](int a, int b) {
                        return weights.weights[a] > weights.weights[b];
                      });

    // Copy top-k indices
    for (int i = 0; i < k; ++i) {
      indices[i] = idx[i];
    }

    // Renormalize probabilities for top-k
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      sum += weights.weights[idx[i]];
    }
    for (int i = 0; i < k; ++i) {
      probs[i] = weights.weights[idx[i]] / sum;
    }
  }

private:
  // =========================================================================
  // Feature computation
  // =========================================================================

  /**
   * Get pre-computed Game Phase from FeatureExtractor globals.
   * 0.0 = Opening, 1.0 = Deep Endgame.
   * (Computed in FeatureExtractor using Material * Mobility formula)
   */
  static float compute_phase(const ChessInput &input) {
    return input.global[FeatureExtractor::GlobalIndices::PHASE];
  }

  static float compute_phase(const FactorizedInput &input) {
    return input.global[FactorizedFeatureExtractor::GlobalIndices::PHASE];
  }

  /**
   * Compute CCT (Checks, Captures, Threats) score.
   * Higher = more tactical.
   */
  static float compute_cct(const ChessInput &input) {
    using L = FeatureExtractor::LayerIndices;

    float cct = 0.0f;

    // ========== CHECKS (Weight: 10) ==========
    bool us_in_check = count_layer(input.layers[L::THEM_CHECKS]) > 0;
    bool them_in_check = count_layer(input.layers[L::US_CHECKS]) > 0;
    cct += (us_in_check ? CCT_WEIGHT_CHECK : 0.0f);
    cct += (them_in_check ? CCT_WEIGHT_CHECK : 0.0f);

    // ========== CAPTURES (Weighted by piece value) ==========
    // US_THREATS shows where we can capture their pieces

    // Count threatened pieces by type
    cct += piece_threat_value(input, L::US_THREATS, L::THEM_PAWN,
                              PIECE_VALUE_PAWN);
    cct += piece_threat_value(input, L::US_THREATS, L::THEM_KNIGHT,
                              PIECE_VALUE_KNIGHT);
    cct += piece_threat_value(input, L::US_THREATS, L::THEM_BISHOP,
                              PIECE_VALUE_BISHOP);
    cct += piece_threat_value(input, L::US_THREATS, L::THEM_ROOK,
                              PIECE_VALUE_ROOK);
    cct += piece_threat_value(input, L::US_THREATS, L::THEM_QUEEN,
                              PIECE_VALUE_QUEEN);

    // THEM_THREATS shows where they can capture our pieces
    cct += piece_threat_value(input, L::THEM_THREATS, L::US_PAWN,
                              PIECE_VALUE_PAWN);
    cct += piece_threat_value(input, L::THEM_THREATS, L::US_KNIGHT,
                              PIECE_VALUE_KNIGHT);
    cct += piece_threat_value(input, L::THEM_THREATS, L::US_BISHOP,
                              PIECE_VALUE_BISHOP);
    cct += piece_threat_value(input, L::THEM_THREATS, L::US_ROOK,
                              PIECE_VALUE_ROOK);
    cct += piece_threat_value(input, L::THEM_THREATS, L::US_QUEEN,
                              PIECE_VALUE_QUEEN);

    // ========== MAJOR THREATS BONUS (Weight: 3) ==========
    bool us_major_threatened =
        has_overlap(input.layers[L::THEM_THREATS], input.layers[L::US_ROOK]) ||
        has_overlap(input.layers[L::THEM_THREATS], input.layers[L::US_QUEEN]);
    bool them_major_threatened =
        has_overlap(input.layers[L::US_THREATS], input.layers[L::THEM_ROOK]) ||
        has_overlap(input.layers[L::US_THREATS], input.layers[L::THEM_QUEEN]);

    cct += (us_major_threatened ? CCT_WEIGHT_MAJOR_THREAT : 0.0f);
    cct += (them_major_threatened ? CCT_WEIGHT_MAJOR_THREAT : 0.0f);

    return cct;
  }

  static bool has_overlap_factorized(const FactorizedInput &input,
                                     int attacker_group_begin,
                                     int attacker_group_end,
                                     int target_presence_group) {
    using FE = FactorizedFeatureExtractor;
    for (int g = attacker_group_begin; g <= attacker_group_end; ++g) {
      for (int sq = 0; sq < 64; ++sq) {
        if (input.branches[g][FE::ATTACKS][sq] > 0.5f &&
            input.branches[target_presence_group][FE::PRESENCE][sq] > 0.5f) {
          return true;
        }
      }
    }
    return false;
  }

  static float compute_cct(const FactorizedInput &input) {
    using FE = FactorizedFeatureExtractor;

    float cct = 0.0f;

    const int us_begin = FE::US_PAWN;
    const int us_end = FE::US_KING;
    const int them_begin = FE::THEM_PAWN;
    const int them_end = FE::THEM_KING;

    bool us_in_check = has_overlap_factorized(input, them_begin, them_end,
                                              FE::US_KING);
    bool them_in_check = has_overlap_factorized(input, us_begin, us_end,
                                                FE::THEM_KING);
    cct += (us_in_check ? CCT_WEIGHT_CHECK : 0.0f);
    cct += (them_in_check ? CCT_WEIGHT_CHECK : 0.0f);

    if (has_overlap_factorized(input, us_begin, us_end, FE::THEM_PAWN))
      cct += PIECE_VALUE_PAWN;
    if (has_overlap_factorized(input, us_begin, us_end,
                               FE::THEM_KNIGHT))
      cct += PIECE_VALUE_KNIGHT;
    if (has_overlap_factorized(input, us_begin, us_end,
                               FE::THEM_BISHOP))
      cct += PIECE_VALUE_BISHOP;
    if (has_overlap_factorized(input, us_begin, us_end, FE::THEM_ROOK))
      cct += PIECE_VALUE_ROOK;
    if (has_overlap_factorized(input, us_begin, us_end,
                               FE::THEM_QUEEN))
      cct += PIECE_VALUE_QUEEN;

    if (has_overlap_factorized(input, them_begin, them_end, FE::US_PAWN))
      cct += PIECE_VALUE_PAWN;
    if (has_overlap_factorized(input, them_begin, them_end,
                               FE::US_KNIGHT))
      cct += PIECE_VALUE_KNIGHT;
    if (has_overlap_factorized(input, them_begin, them_end,
                               FE::US_BISHOP))
      cct += PIECE_VALUE_BISHOP;
    if (has_overlap_factorized(input, them_begin, them_end, FE::US_ROOK))
      cct += PIECE_VALUE_ROOK;
    if (has_overlap_factorized(input, them_begin, them_end,
                               FE::US_QUEEN))
      cct += PIECE_VALUE_QUEEN;

    bool us_major_threatened =
        has_overlap_factorized(input, them_begin, them_end, FE::US_ROOK) ||
        has_overlap_factorized(input, them_begin, them_end,
                               FE::US_QUEEN);
    bool them_major_threatened =
        has_overlap_factorized(input, us_begin, us_end, FE::THEM_ROOK) ||
        has_overlap_factorized(input, us_begin, us_end,
                               FE::THEM_QUEEN);

    cct += (us_major_threatened ? CCT_WEIGHT_MAJOR_THREAT : 0.0f);
    cct += (them_major_threatened ? CCT_WEIGHT_MAJOR_THREAT : 0.0f);

    return cct;
  }

  /**
   * Check if position has major pieces (rooks or queens)
   */
  static bool has_major_pieces(const ChessInput &input) {
    using G = FeatureExtractor::GlobalIndices;
    return input.global[G::US_MAT_ROOK] > 0.0f ||
           input.global[G::THEM_MAT_ROOK] > 0.0f ||
           input.global[G::US_MAT_QUEEN] > 0.0f ||
           input.global[G::THEM_MAT_QUEEN] > 0.0f;
  }

  /**
   * Compute major piece ratio for endgame blending.
   *
   * Returns 0.0 for pure minor endgame (no rooks/queens)
   * Returns 1.0 for major-heavy endgame (lots of rooks/queens)
   * Intermediate values for mixed endgames
   *
   * Formula: major_material / (major_material + minor_material)
   */
  static float compute_major_ratio(const ChessInput &input) {
    using G = FeatureExtractor::GlobalIndices;

    // Calculate major piece material (rooks + queens)
    float major_mat = 0.0f;
    major_mat +=
        (input.global[G::US_MAT_ROOK] + input.global[G::THEM_MAT_ROOK]) * 2.0f *
        PIECE_VALUE_ROOK;
    major_mat +=
        (input.global[G::US_MAT_QUEEN] + input.global[G::THEM_MAT_QUEEN]) *
        1.0f * PIECE_VALUE_QUEEN;

    // Calculate minor piece material (knights + bishops)
    float minor_mat = 0.0f;
    minor_mat +=
        (input.global[G::US_MAT_KNIGHT] + input.global[G::THEM_MAT_KNIGHT]) *
        2.0f * PIECE_VALUE_KNIGHT;
    minor_mat +=
        (input.global[G::US_MAT_BISHOP] + input.global[G::THEM_MAT_BISHOP]) *
        2.0f * PIECE_VALUE_BISHOP;

    float total = major_mat + minor_mat;
    if (total < 1e-6f) {
      return 0.5f; // Pawn endgame - neutral
    }

    return major_mat / total;
  }

  static float compute_major_ratio(const FactorizedInput &input) {
    using G = FactorizedFeatureExtractor::GlobalIndices;

    float major_mat = 0.0f;
    major_mat +=
        (input.global[G::US_MAT_ROOK] + input.global[G::THEM_MAT_ROOK]) * 2.0f *
        PIECE_VALUE_ROOK;
    major_mat +=
        (input.global[G::US_MAT_QUEEN] + input.global[G::THEM_MAT_QUEEN]) *
        1.0f * PIECE_VALUE_QUEEN;

    float minor_mat = 0.0f;
    minor_mat +=
        (input.global[G::US_MAT_KNIGHT] + input.global[G::THEM_MAT_KNIGHT]) *
        2.0f * PIECE_VALUE_KNIGHT;
    minor_mat +=
        (input.global[G::US_MAT_BISHOP] + input.global[G::THEM_MAT_BISHOP]) *
        2.0f * PIECE_VALUE_BISHOP;

    float total = major_mat + minor_mat;
    if (total < 1e-6f) {
      return 0.5f;
    }

    return major_mat / total;
  }

  /**
   * Compute material imbalance between sides (in pawn units)
   */
  static float compute_material_imbalance(const ChessInput &input) {
    using G = FeatureExtractor::GlobalIndices;

    float us_mat = 0.0f;
    us_mat += input.global[G::US_MAT_PAWN] * 8.0f * 1.0f;
    us_mat += input.global[G::US_MAT_KNIGHT] * 2.0f * 3.0f;
    us_mat += input.global[G::US_MAT_BISHOP] * 2.0f * 3.0f;
    us_mat += input.global[G::US_MAT_ROOK] * 2.0f * 5.0f;
    us_mat += input.global[G::US_MAT_QUEEN] * 1.0f * 9.0f;

    float them_mat = 0.0f;
    them_mat += input.global[G::THEM_MAT_PAWN] * 8.0f * 1.0f;
    them_mat += input.global[G::THEM_MAT_KNIGHT] * 2.0f * 3.0f;
    them_mat += input.global[G::THEM_MAT_BISHOP] * 2.0f * 3.0f;
    them_mat += input.global[G::THEM_MAT_ROOK] * 2.0f * 5.0f;
    them_mat += input.global[G::THEM_MAT_QUEEN] * 1.0f * 9.0f;

    return std::abs(us_mat - them_mat);
  }

  // =========================================================================
  // Utility functions
  // =========================================================================

  /**
   * Count number of set squares in a layer (value > 0)
   * Optimized using 64-bit strides.
   */
  static int count_layer(const uint8_t layer[64]) {
    const uint64_t *ptr = reinterpret_cast<const uint64_t *>(layer);
    int count = 0;
    for (int i = 0; i < 8; ++i) {
      uint64_t val = ptr[i];
      if (val) {
        // Each byte is either 0x00 or 0xFF.
        // We can just count the set bytes.
        // Simple way: iterate bytes if the 64-block is non-zero
        const uint8_t *b = reinterpret_cast<const uint8_t *>(&val);
        for (int j = 0; j < 8; ++j)
          if (b[j])
            count++;
      }
    }
    return count;
  }

  /**
   * Check if two layers have any overlap (both non-zero at same square)
   * Optimized using 64-bit strides.
   */
  static bool has_overlap(const uint8_t a[64], const uint8_t b[64]) {
    const uint64_t *a64 = reinterpret_cast<const uint64_t *>(a);
    const uint64_t *b64 = reinterpret_cast<const uint64_t *>(b);

    // Unrolled loop for speed
    if (a64[0] & b64[0])
      return true;
    if (a64[1] & b64[1])
      return true;
    if (a64[2] & b64[2])
      return true;
    if (a64[3] & b64[3])
      return true;
    if (a64[4] & b64[4])
      return true;
    if (a64[5] & b64[5])
      return true;
    if (a64[6] & b64[6])
      return true;
    if (a64[7] & b64[7])
      return true;

    return false;
  }

  /**
   * Get threat value for pieces of a specific type
   */
  static float piece_threat_value(const ChessInput &input, int threat_layer,
                                  int piece_layer, float piece_value) {
    if (has_overlap(input.layers[threat_layer], input.layers[piece_layer])) {
      return piece_value;
    }
    return 0.0f;
  }

  // =========================================================================
  // Math Helpers
  // =========================================================================

  /**
   * Fast approximation of exp(x).
   * Range: Valid for inputs approx -87 to +87.
   * Speed: ~10x faster than std::exp.
   */
  static inline float fast_exp(float x) {
    // Clamp to avoid Infinity/NaN artifacts
    if (x < -88.0f)
      return 0.0f;
    if (x > 88.0f)
      x = 88.0f;

    // Schraudolph's approximation / Magic Polynomial
    x = 1.0f + x / 256.0f;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    return x;
  }

  /**
   * Fast sigmoid approximation using fast_exp.
   * sigmoid(x) = 1 / (1 + exp(-x))
   * Range: 0.0 to 1.0 (centered at 0.0)
   */
  static float sigmoid(float x) {
    if (x > 10.0f)
      return 1.0f;
    if (x < -10.0f)
      return 0.0f;
    return 1.0f / (1.0f + fast_exp(-x));
  }

  /**
   * Optimized Softmax with Safety Fallback.
   * If numerical instability occurs, defaults to TACTICAL/STRATEGIC split.
   */
  static void softmax(const float *scores, float *probs, int n) {
    // 1. Find Max (Shift invariance to prevent overflow)
    float max_score = scores[0];
    for (int i = 1; i < n; ++i) {
      if (scores[i] > max_score)
        max_score = scores[i];
    }

    // 2. Exponentiate & Sum
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
      // fast_exp guarantees result >= 0
      float e = fast_exp(scores[i] - max_score);
      probs[i] = e;
      sum += e;
    }

    // 3. SAFETY CHECK
    // If sum is NaN or effectively zero (should be impossible with max-shift,
    // but hardware glitches or NaN inputs can cause it), fallback to safety.
    if (sum < 1e-9f || std::isnan(sum)) {
      // Reset all to 0
      for (int i = 0; i < n; ++i)
        probs[i] = 0.0f;

      // Default: 50% Tactical, 50% Strategic
      // Assumes TACTICAL=0, STRATEGIC=1 as defined in enum
      if (n >= 2) {
        probs[0] = 0.5f;
        probs[1] = 0.5f;
      } else {
        probs[0] = 1.0f;
      }
      return;
    }

    // 4. Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
      probs[i] *= inv_sum;
    }
  }
};
