#pragma once

/**
 * @file SearchHeuristics.h
 * @brief Search data structures.
 *
 * Consolidates large multi-dimensional arrays used for search heuristics
 * (killers, history, continuation history) into a single struct suitable
 * for heap allocation, avoiding stack overflow issues.
 */

#include <array>
#include <cstdint>
#include "ISearch.h"

/**
 * @struct SearchHeuristics
 * @brief Container for stateful search heuristics and move ordering tables.
 *
 * These arrays are typically too large to allocate safely on the thread stack,
 * so they are grouped here to be dynamically allocated by the search algorithm.
 */
struct SearchHeuristics {
  std::array<std::array<Move, 2>, SearchConstants::MAX_PLY> killers_{}; ///< Killer moves for each ply (up to 2 per ply).
  std::array<std::array<int, 64>, 64> history_{};                       ///< Standard history heuristic [fromSq][toSq].
  std::array<std::array<Move, 64>, 64> countermove_{};                  ///< Countermove heuristic [fromSq][toSq].
  std::array<std::array<std::array<std::array<int, 64>, 16>, 64>, 16> contHist_{}; ///< Continuation history [prevPiece][prevToSq][piece][toSq].
  std::array<std::array<int, 64>, 16> captureHist_{};                   ///< Capture history [piece][toSq].
  
  std::array<std::array<int16_t, 16384>, 2> pawnCorrHist_{};            ///< Pawn structure correction history [color][pawnHash].
  std::array<std::array<std::array<int16_t, 16384>, 2>, 2> nonPawnCorrHist_{}; ///< Non-pawn correction history [color][type][nonPawnHash].
  std::array<std::array<std::array<std::array<int16_t, 64>, 16>, 64>, 16> contCorrHist_{}; ///< Continuation correction history.

  std::array<std::array<Move, SearchConstants::MAX_PLY>, SearchConstants::MAX_PLY> pvTable_{}; ///< Principal Variation table.
  std::array<int, SearchConstants::MAX_PLY> failHighCount_{};           ///< Count of fail-highs per ply.

  // Pre-allocated move generation buffers per ply to avoid vector allocations
  std::array<std::array<Move, 256>, SearchConstants::MAX_PLY> captures_{};      ///< Buffer for generated capture moves.
  std::array<std::array<int, 256>, SearchConstants::MAX_PLY> captureScores_{};  ///< Scores corresponding to capture moves.
  std::array<std::array<Move, 256>, SearchConstants::MAX_PLY> quiets_{};        ///< Buffer for generated quiet moves.
  std::array<std::array<int, 256>, SearchConstants::MAX_PLY> quietScores_{};    ///< Scores corresponding to quiet moves.
  std::array<std::array<Move, 256>, SearchConstants::MAX_PLY> badCaptures_{};   ///< Buffer for captures that fail SEE.
  std::array<std::array<Move, 256>, SearchConstants::MAX_PLY> searchedQuiets_{}; ///< Buffer for tracking which quiets have been searched.
};
