#pragma once

/**
 * @file Tablebase.h
 * @brief Syzygy tablebase probing via Fathom.
 *
 * Provides WDL (Win-Draw-Loss) and DTZ (Distance to Zero) lookups 
 * for endgame positions using the Fathom library.
 */

#include "../Types.h"
#include <string>
#include <optional>

namespace Tablebase {

/**
 * @enum WDL
 * @brief Tablebase probe result for Win-Draw-Loss.
 */
enum class WDL {
    WIN = 2,            ///< Forced win.
    CURSED_WIN = 1,     ///< Win, but thwarted by the 50-move rule (draw).
    DRAW = 0,           ///< Forced draw.
    BLESSED_LOSS = -1,  ///< Loss, but saved by the 50-move rule (draw).
    LOSS = -2           ///< Forced loss.
};

/**
 * @brief Initializes tablebases with the given path.
 * 
 * @param path Path to the Syzygy tablebase files.
 * @return True on success, false otherwise.
 */
bool init(const std::string& path);

/**
 * @brief Checks if a position is within the tablebase piece limits.
 * 
 * @param board The chess board to check.
 * @return True if the board has <= maxPieces() pieces.
 */
bool available(const Board& board);

/**
 * @brief Probes the Win-Draw-Loss (WDL) table.
 * 
 * @param board The chess board to probe.
 * @return WDL value from side-to-move's perspective, or std::nullopt on failure.
 */
std::optional<WDL> probeWDL(const Board& board);

/**
 * @brief Probes the Distance-to-Zero (DTZ) table for root moves.
 * 
 * @param board The chess board to probe.
 * @return Tuple of best move, WDL value, and DTZ value, or std::nullopt on failure.
 */
std::optional<std::tuple<Move, WDL, int>> probeRoot(const Board& board);

/**
 * @brief Converts a WDL result to a centipawn score for search evaluation.
 * 
 * @param wdl The WDL result.
 * @param mate_ply Distance to mate (used to scale WIN/LOSS scores).
 * @return Centipawn score representation of the tablebase result.
 */
int wdlToScore(WDL wdl, int mate_ply);

/**
 * @brief Gets the maximum number of pieces supported by the loaded tablebases.
 */
int maxPieces();

/**
 * @brief Checks if tablebases have been successfully initialized.
 */
bool isInitialized();

} // namespace Tablebase
