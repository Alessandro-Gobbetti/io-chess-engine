#pragma once

/**
 * @file Types.h
 * @brief Common type definitions and constants for the chess engine.
 * 
 * This file wraps the underlying `chess.hpp` types into convenient aliases
 * and defines globally used constants for search, move ordering, and evaluation.
 */

#include "chess.hpp"

// Type aliases for cleaner code
using Board = chess::Board;         ///< Alias for chess::Board
using Move = chess::Move;           ///< Alias for chess::Move
using Movelist = chess::Movelist;   ///< Alias for chess::Movelist
using Color = chess::Color;         ///< Alias for chess::Color
using Square = chess::Square;       ///< Alias for chess::Square
using Piece = chess::Piece;         ///< Alias for chess::Piece
using PieceType = chess::PieceType; ///< Alias for chess::PieceType
using Bitboard = chess::Bitboard;   ///< Alias for chess::Bitboard
using File = chess::File;           ///< Alias for chess::File
using Rank = chess::Rank;           ///< Alias for chess::Rank

/**
 * @namespace SearchConstants
 * @brief Global constants related to search and evaluation boundaries.
 * 
 * NOTE: All score constants must fit in int16_t (max 32767) for TT storage.
 * After value_to_tt adds ply (up to MAX_PLY=128), max stored value is ~31628.
 * Hierarchy: INFINITE > MATE_SCORE > MATE_IN_MAX > TB_WIN_SCORE > TB_WIN_IN_MAX > WDL eval (≤30000)
 */
namespace SearchConstants {
constexpr int INFINITE = 32000;                      ///< Absolute maximum score.
constexpr int MATE_SCORE = 31500;                    ///< Base score for checkmate.
constexpr int MATE_IN_MAX = MATE_SCORE - 300;        ///< Threshold to detect mating sequences.
constexpr int TB_WIN_SCORE = 31000;                  ///< Base score for a tablebase win.
constexpr int TB_WIN_IN_MAX = TB_WIN_SCORE - 800;    ///< Threshold to detect tablebase wins.
constexpr int DRAW_SCORE = 0;                        ///< Score for a draw position.
constexpr int CONTEMPT_FACTOR = 5;                   ///< Make draws slightly negative to encourage winning.

// TT Flags
constexpr uint8_t TT_EXACT = 0; ///< Transposition table entry is an exact score.
constexpr uint8_t TT_ALPHA = 1; ///< Transposition table entry is an upper bound (failed low).
constexpr uint8_t TT_BETA = 2;  ///< Transposition table entry is a lower bound (failed high).

// Search defaults
constexpr int DEFAULT_DEPTH = 100;       ///< Default maximum depth for search.
constexpr int MAX_PLY = 128;             ///< Maximum supported plies from root.
constexpr int LMR_MIN_DEPTH = 3;         ///< Minimum depth to consider Late Move Reductions (LMR).
constexpr int NULL_MOVE_MIN_DEPTH = 4;   ///< Minimum depth to consider Null Move Pruning.

// Extension limits
constexpr int MAX_EXTENSIONS = 8;        ///< Maximum number of search extensions per path.

// Sentinel for invalid static eval
constexpr int16_t INVALID_EVAL = 32001;  ///< Special value indicating an uncomputed or invalid static evaluation.
} // namespace SearchConstants

/**
 * @namespace MoveOrderConstants
 * @brief Scores used to sort moves during the move picking phase.
 */
namespace MoveOrderConstants {
constexpr int16_t HASH_MOVE_SCORE = 30000; ///< Highest priority for moves retrieved from the Transposition Table.
constexpr int16_t CAPTURE_BASE = 10000;    ///< Base score for captures, adjusted by MVV-LVA.
constexpr int16_t KILLER_SCORE_1 = 9000;   ///< Score for the primary killer move.
constexpr int16_t KILLER_SCORE_2 = 8000;   ///< Score for the secondary killer move.
constexpr int16_t HISTORY_MAX = 8000;      ///< Maximum possible history heuristic score.
} // namespace MoveOrderConstants

/**
 * @namespace MoveGen
 * @brief Utility wrappers for legal and capture move generation.
 */
namespace MoveGen {

/**
 * @brief Generates all pseudo-legal and legal moves for the given board state.
 * 
 * @param moves List to append the generated moves to.
 * @param board Current board state.
 */
inline void generateLegalMoves(Movelist &moves, const Board &board) {
  chess::movegen::legalmoves(moves, board);
}

/**
 * @brief Generates only legal capture and promotion moves for the given board state.
 * 
 * @param moves List to append the generated capture moves to.
 * @param board Current board state.
 */
inline void generateCaptures(Movelist &moves, const Board &board) {
  chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves, board);
}
} // namespace MoveGen
