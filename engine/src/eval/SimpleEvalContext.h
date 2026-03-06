#pragma once
#include "FeatureExtractor.hpp"
#include "../Types.h"
#include "IEvaluator.h"
#include <algorithm>
#include <array>

struct EvalScore {
  int mg;
  int eg;
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

  static int mirror(int sq) { return sq ^ 56; }

  Score evalOneSide(const Board &board, Color us, int &phase) const;
  // FIXED: Synchronized with .cpp signature
  void evalPawns(const Board &board, Color us, const Bitboard &ourPawns,
                 const Bitboard &theirPawns, Score &score) const;
  void evalKingSafety(const Board &board, Color us, const Bitboard &ourPawns,
                      const Bitboard &theirPawns, Score &score) const;

public:
  SimpleEvalContext() = default;
  // Standard interface (will compute features internally)
  float evaluate(const Board &board) override;
};