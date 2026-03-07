#pragma once
// Tablebase.h - Syzygy tablebase probing via Fathom
// Provides WDL (Win-Draw-Loss) and DTZ (Distance to Zero) lookups

#include "../Types.h"
#include <string>
#include <optional>

namespace Tablebase {

// Tablebase probe result
enum class WDL {
    WIN = 2,
    CURSED_WIN = 1,  // Win but 50-move rule draw
    DRAW = 0,
    BLESSED_LOSS = -1,  // Loss but 50-move rule draw
    LOSS = -2
};

// Initialize tablebases with given path
// Returns true on success
bool init(const std::string& path);

// Check if position is in tablebase range (inline for performance)
// Returns true if <=5 pieces (or configured max)
bool available(const Board& board);

// Probe Win-Draw-Loss (WDL) table
// Returns WDL value from side-to-move perspective
// Returns nullopt if not available or probe fails
std::optional<WDL> probeWDL(const Board& board);

// Probe Distance-to-Zero (DTZ) table for root moves
// Returns best move, WDL value, and DTZ value
// Returns nullopt if not available or probe fails
std::optional<std::tuple<Move, WDL, int>> probeRoot(const Board& board);

// Convert WDL to centipawn score for search
// mate_ply is distance to mate (used for WIN/LOSS)
int wdlToScore(WDL wdl, int mate_ply);

// Get maximum pieces supported
int maxPieces();

// Check if tablebases are initialized
bool isInitialized();

} // namespace Tablebase
