/**
 * @file FactorizedFeatureExtractor.hpp
 * @brief Feature extraction logic for generating packed factorized features.
 */
#pragma once
#include "chess.hpp"
#include <array>
#include <cstdint>

// ============================================================================
// FactorizedInput — Output struct for ChessNetFactorizedMoE
// ============================================================================
//
// Layout (all float, no uint8→float conversion needed at inference time):
//
//   branches[12][MAX_BRANCH_PLANES][64]
//       12 piece groups × up to 10 branch planes × 64 squares.
//       Each piece type uses a different number of planes (see below);
//       unused trailing planes are always zero.
//
//   bypass[12][64]       — Global heatmaps + volatile tactical maps
//   global[32]           — Scalar features (Material, Flags, etc.)
//
// Piece-group ordering (side-to-move relative):
//
//   0  US Pawns    6  THEM Pawns
//   1  US Knights  7  THEM Knights
//   2  US Bishops  8  THEM Bishops
//   3  US Rooks    9  THEM Rooks
//   4  US Queens  10  THEM Queens
//   5  US King    11  THEM King
//
// Legacy WHITE/BLACK enum names are kept for compatibility, but at runtime
// these groups are filled relative to side-to-move.
//
// Per-group planes vary by piece type:
//
//   Pawns   (4): Presence, Mobility, Attacks, Defends
//   Knights (4): Presence, Mobility, Attacks, Defends
//   Bishops (5): Presence, Mobility, Attacks, Defends, X-Ray
//   Rooks   (5): Presence, Mobility, Attacks, Defends, X-Ray
//   Queens   (5): Presence, Mobility, Attacks, Defends, X-Ray
//   Kings    (4): Presence, Mobility, Attacks, Defends
//
// ============================================================================

struct alignas(64) FactorizedInput {
  static constexpr int MAX_BRANCH_PLANES = 10;  // Logical max across supported schemas

  float branches[12][MAX_BRANCH_PLANES][64]; // 12 groups × up to 10 planes × 64 squares
  float bypass[12][64];      // 12 bypass planes
  float global[32];          // 32 scalars
};

class FactorizedFeatureExtractor {
public:
  // --- Branch group indices ---
  enum BranchIndex {
    US_PAWN = 0,
    US_KNIGHT,
    US_BISHOP,
    US_ROOK,
    US_QUEEN,
    US_KING,
    THEM_PAWN,
    THEM_KNIGHT,
    THEM_BISHOP,
    THEM_ROOK,
    THEM_QUEEN,
    THEM_KING,
  };

  // --- Common sub-plane indices (shared across all piece types) ---
  enum PlaneIndex {
    PRESENCE = 0,
    MOBILITY = 1,
    ATTACKS = 2,
    DEFENDS = 3,
    X_RAY = 4,
  };

  static constexpr int NUM_PAWN_PLANES = 4;
  static constexpr int NUM_KNIGHT_PLANES = 4;
  static constexpr int NUM_BISHOP_PLANES = 5;
  static constexpr int NUM_ROOK_PLANES = 5;

  // Queens (5 planes total — unchanged)
  static constexpr int NUM_QUEEN_PLANES = 5;

  // Kings (4 planes total)
  static constexpr int NUM_KING_PLANES = 4;

  // Plane count per piece type (indexed by piece_offset 0-5)
  static constexpr int PLANES_PER_TYPE[6] = {
    NUM_PAWN_PLANES,    // 0: Pawn   = 4
    NUM_KNIGHT_PLANES,  // 1: Knight = 4
    NUM_BISHOP_PLANES,  // 2: Bishop = 5
    NUM_ROOK_PLANES,    // 3: Rook   = 5
    NUM_QUEEN_PLANES,   // 4: Queen  =  5
    NUM_KING_PLANES,    // 5: King   = 4
  };

  static constexpr int NUM_BRANCH_PLANES_RICH = FactorizedInput::MAX_BRANCH_PLANES;

  // --- Bypass plane indices ---
  enum BypassIndex {
    US_KING_DIST = 0,
    THEM_KING_DIST = 1,
    US_PAWN_ATTACKS = 2,
    THEM_PAWN_ATTACKS = 3,
    TOTAL_CONTROL = 4,
    ABSOLUTE_PINS = 5,
    EN_PRISE = 6,
    SAFE_MOBILITY_MAP = 7,
    KING_RING_MAP = 8,
    PASSED_PAWN_MAP = 9,
    OUTPOST_MAP = 10,
    OPEN_LINES_MAP = 11,
  };

  static constexpr int NUM_BYPASS_PLANES = 12;
  static_assert(OPEN_LINES_MAP < NUM_BYPASS_PLANES,
                "Bypass index exceeds bypass plane storage");

  // --- Global scalar indices (identical to FeatureExtractor) ---
  enum GlobalIndices {
    US_MAT_PAWN = 0,
    THEM_MAT_PAWN,
    US_MAT_KNIGHT,
    THEM_MAT_KNIGHT,
    US_MAT_BISHOP,
    THEM_MAT_BISHOP,
    US_MAT_ROOK,
    THEM_MAT_ROOK,
    US_MAT_QUEEN,
    THEM_MAT_QUEEN,
    US_OO,
    US_OOO,
    THEM_OO,
    THEM_OOO,
    PHASE,
    US_BISHOP_PAIR,
    THEM_BISHOP_PAIR,
    OPPOSITE_BISHOPS,
    US_ROOK_VS_MINORS,
    THEM_ROOK_VS_MINORS,
    IS_WHITE_TO_MOVE,
  };

  // --- Main entry points ---
  static void fill_input(const chess::Board &board, FactorizedInput &out);
  static void fill_input_rich(const chess::Board &board, FactorizedInput &out);
};
