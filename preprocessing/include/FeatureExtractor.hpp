/**
 * @file FeatureExtractor.hpp
 * @brief Standard sparse/dense feature extraction logic for chess positions.
 */
#pragma once
#include "chess.hpp"
#include <array>
#include <cstdint>

struct alignas(64) ChessInput {
  // [Input 1] Board: 32 planes of 8x8 (Categorical: 0 or 255, or continuous
  // [last two planes]) Structure: 32 x 64 bytes = 2048 bytes
  uint8_t layers[32][64];

  // [Input 2] Global: 16 scalars (Normalized Floats)
  // Structure: 16 x 4 bytes = 64 bytes (aligned)
  float global[16];
};

class FeatureExtractor {
public:
  static constexpr int NUM_SQUARES = 64;

  // --- Sparse Layer Indices (0 to 31) ---
  // Maps to ChessInput.sparse[index]
  enum LayerIndices {
    // Piece Presence (0-11)
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

    // Legal Moves / Reachability (12-23)
    US_MOVE_PAWN,
    US_MOVE_KNIGHT,
    US_MOVE_BISHOP,
    US_MOVE_ROOK,
    US_MOVE_QUEEN,
    US_MOVE_KING,
    THEM_MOVE_PAWN,
    THEM_MOVE_KNIGHT,
    THEM_MOVE_BISHOP,
    THEM_MOVE_ROOK,
    THEM_MOVE_QUEEN,
    THEM_MOVE_KING,

    // Threats & Checks (24-27)
    US_THREATS,
    THEM_THREATS,
    US_CHECKS,
    THEM_CHECKS,

    // Defended (28-29)
    US_DEFENDED,
    THEM_DEFENDED,

    // Special (30-31) - Added to reach exactly 32 Channels for alignment
    US_KING_DIST,
    THEM_KING_DIST
  };

  // --- Global Indices (0 to 14) ---
  // Maps to ChessInput.global[index]
  enum GlobalIndices {
    // Material Configuration (Normalized)
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

    // Castling Rights
    US_OO,
    US_OOO,
    THEM_OO,
    THEM_OOO,

    // Game Phase (0.0 = Opening, 1.0 = Deep Endgame)
    PHASE,
  };

  // High-performance extraction directly into the struct
  static void fill_input(const chess::Board &board, ChessInput &out);

private:
  // Internal logic for dense map generation (Chebyshev distance)
  // Implementation is hidden in .cpp to keep header clean
  static void init_tables();
};
