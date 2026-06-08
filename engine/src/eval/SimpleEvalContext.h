#pragma once

/**
 * @file SimpleEvalContext.h
 * @brief Hand-crafted evaluation function for testing and fallback.
 *
 * Provides a classical evaluation function utilizing material counting,
 * Piece-Square Tables (PSQT), pawn structure evaluation, and king safety.
 */

#include "FeatureExtractor.hpp"
#include "../Types.h"
#include "IEvaluator.h"
#include <algorithm>
#include <array>

/**
 * @struct EvalScore
 * @brief Holds separate middle-game (mg) and end-game (eg) scores.
 *
 * Used for tapered evaluation to interpolate smoothly between game phases.
 */
struct EvalScore {
  int mg; ///< Middle-game evaluation score.
  int eg; ///< End-game evaluation score.
  constexpr EvalScore(int m = 0, int e = 0) : mg(m), eg(e) {}
  EvalScore &operator+=(const EvalScore &rhs) {
    mg += rhs.mg;
    eg += rhs.eg;
    return *this;
  }
  EvalScore &operator-=(const EvalScore &rhs) {
    mg -= rhs.mg;
    eg -= rhs.eg;
    return *this;
  }
  EvalScore operator+(const EvalScore &rhs) const {
    return EvalScore(mg + rhs.mg, eg + rhs.eg);
  }
  EvalScore operator-(const EvalScore &rhs) const {
    return EvalScore(mg - rhs.mg, eg - rhs.eg);
  }
};

/**
 * @class SimpleEvalContext
 * @brief Classical, non-neural evaluation context.
 *
 * Implements a traditional evaluation function based on material, PSQT,
 * pawn structure, mobility, and king safety. Useful as a fallback when
 * the neural network is unavailable.
 */
class SimpleEvalContext : public IEvaluator {
public:
  using Score = EvalScore;

  // Made public so the .cpp can see them easily
  static constexpr int PHASE_KNIGHT = 1;
  static constexpr int PHASE_BISHOP = 1;
  static constexpr int PHASE_ROOK = 2;
  static constexpr int PHASE_QUEEN = 4;
  static constexpr int TOTAL_PHASE = 24;
  static constexpr int TEMPO_BONUS = 15;

  static const Score PIECE_VALUES[6];
  static const Score PSQT[6][64];
  static constexpr Score BISHOP_PAIR = {30, 80};
  static constexpr Score ROOK_OPEN_FILE = {35, 15};
  static constexpr Score ROOK_SEMI_OPEN = {15, 10};
  static constexpr Score PAWN_PASSED[8] = {{0, 0},     {10, 10}, {10, 20},
                                           {20, 40},   {40, 80}, {80, 160},
                                           {150, 250}, {0, 0}};
  static constexpr Score PAWN_ISOLATED = {-10, -15};
  static constexpr Score PAWN_DOUBLED = {-15, -20};

  /**
   * @brief Mirrors a square index vertically (for black's perspective).
   */
  static int mirror(int sq) { return sq ^ 56; }

  /**
   * @brief Evaluates features specific to one side.
   */
  Score evalOneSide(const Board &board, Color us, int &phase) const;
  
  /**
   * @brief Evaluates pawn structure features.
   */
  void evalPawns(const Board &board, Color us, const Bitboard &ourPawns,
                 const Bitboard &theirPawns, Score &score) const;
                 
  /**
   * @brief Evaluates king safety and pawn shield quality.
   */
  void evalKingSafety(const Board &board, Color us, const Bitboard &ourPawns,
                      const Bitboard &theirPawns, Score &score) const;

public:
  SimpleEvalContext() = default;
  
  /**
   * @brief Computes the tapered static evaluation of the board.
   * 
   * @param board The chess board position.
   * @param ply Distance from the search root.
   * @return A scalar score from the perspective of the side to move.
   */
  float evaluate(const Board &board, int ply = 0) override;
};