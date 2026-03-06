#pragma once
// Types.h - Common type definitions for the chess engine
// Wraps chess.hpp types for convenient usage

#include "chess.hpp"

// Type aliases for cleaner code
using Board = chess::Board;
using Move = chess::Move;
using Movelist = chess::Movelist;
using Color = chess::Color;
using Square = chess::Square;
using Piece = chess::Piece;
using PieceType = chess::PieceType;
using Bitboard = chess::Bitboard;
using File = chess::File;
using Rank = chess::Rank;

// Search Constants
// NOTE: All score constants must fit in int16_t (max 32767) for TT storage.
// After value_to_tt adds ply (up to MAX_PLY=128), max stored value is ~31628.
// Hierarchy: INFINITE > MATE_SCORE > MATE_IN_MAX > TB_WIN_SCORE > TB_WIN_IN_MAX > WDL eval (≤30000)
namespace SearchConstants {
constexpr int INFINITE = 32000;
constexpr int MATE_SCORE = 31500;
constexpr int MATE_IN_MAX = MATE_SCORE - 300;
constexpr int TB_WIN_SCORE = 31000;
constexpr int TB_WIN_IN_MAX = TB_WIN_SCORE - 800;
constexpr int DRAW_SCORE = 0;
constexpr int CONTEMPT_FACTOR =
    5; // Make draws slightly negative to encourage winning

// TT Flags
constexpr uint8_t TT_EXACT = 0;
constexpr uint8_t TT_ALPHA = 1; // Upper bound (failed low)
constexpr uint8_t TT_BETA = 2;  // Lower bound (failed high)

// Search defaults
constexpr int DEFAULT_DEPTH = 100;
constexpr int MAX_PLY = 128;
constexpr int LMR_THRESHOLD = 4; // DEPRECATED: Moved to TunableParams
constexpr int LMR_MIN_DEPTH = 3; // Minimum depth for LMR
constexpr int NULL_MOVE_R = 3;   // DEPRECATED: Moved to TunableParams
constexpr int NULL_MOVE_MIN_DEPTH = 4;
constexpr int FUTILITY_MARGIN = 200; // DEPRECATED: Moved to TunableParams
constexpr int DELTA_MARGIN = 200;    // DEPRECATED: Moved to TunableParams

// Extension limits
constexpr int MAX_EXTENSIONS = 8; // Maximum extensions per path

// Razoring margins (tuned) - drop to qsearch if eval is far below alpha
// These should be roughly: "max positional gain possible in N plies"
constexpr int RAZOR_MARGIN_D1 = 350; // DEPRECATED: Moved to TunableParams
constexpr int RAZOR_MARGIN_D2 = 700; // DEPRECATED: Moved to TunableParams
constexpr int RAZOR_MARGIN_D3 = 900; // DEPRECATED: Moved to TunableParams

// Reverse Futility Pruning (static null move pruning)
constexpr int REVERSE_FUTILITY_MARGIN =
    120; // DEPRECATED: Moved to TunableParams

// Late Move Pruning thresholds by depth
constexpr int LMP_BASE = 2; // DEPRECATED: Moved to TunableParams

// Singular extension
// Singular extension
constexpr int16_t INVALID_EVAL = 32001; // Sentinel for invalid static eval
} // namespace SearchConstants

// Move ordering scores
namespace MoveOrderConstants {
constexpr int16_t HASH_MOVE_SCORE = 30000;
constexpr int16_t CAPTURE_BASE = 10000;
constexpr int16_t KILLER_SCORE_1 = 9000;
constexpr int16_t KILLER_SCORE_2 = 8000;
constexpr int16_t HISTORY_MAX = 8000;
} // namespace MoveOrderConstants

// Helper functions for move generation
namespace MoveGen {
inline void generateLegalMoves(Movelist &moves, const Board &board) {
  chess::movegen::legalmoves(moves, board);
}

inline void generateCaptures(Movelist &moves, const Board &board) {
  chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves,
                                                                   board);
}
} // namespace MoveGen
