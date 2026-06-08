#pragma once

/**
 * @file MovePicker.h
 * @brief Move selection and ordering for the search algorithm.
 *
 * Implements a phased move picker that lazily generates, scores, and selects
 * moves to maximize Alpha-Beta cutoffs and reduce overall search time.
 */

#include "ISearch.h"
#include <array>

// Forward declarations
class Negamax;

/**
 * @class MovePicker
 * @brief Lazily selects the best move at each step of the search.
 *
 * Avoids scoring and sorting all moves upfront, which is crucial for 
 * cutoff-heavy nodes. Uses a phased approach: Transposition Table move, 
 * Queen Promotions, Good Captures (SEE > 0), Killer Moves, Countermoves, 
 * Quiet Moves, and finally Bad Captures (SEE < 0).
 */
class MovePicker {
private:
  const Movelist &allMoves_;   ///< Reference to all legal moves generated for the current node.
  
  // === OPTIMIZATION: Fixed-size arrays to avoid dynamic allocations ===
  // Max legal moves in any chess position is 218, but typical positions have <50
  static constexpr int MAX_MOVES = 256; ///< Maximum allowed moves, aligned to power of 2 for safety.
  
  std::array<Move, MAX_MOVES> &captures_;      ///< External buffer for capture moves.
  std::array<int, MAX_MOVES> &captureScores_;  ///< External buffer for capture scores.
  int captureCount_ = 0;                       ///< Number of captures populated.
  
  std::array<Move, 2> killerMoves_;            ///< Local cache of killer moves for this ply.
  int killerCount_ = 0;                        ///< Number of valid killer moves found.
  
  std::array<Move, MAX_MOVES> &quiets_;        ///< External buffer for quiet moves.
  std::array<int, MAX_MOVES> &quietScores_;    ///< External buffer for quiet scores.
  int quietCount_ = 0;                         ///< Number of quiets populated.
  
  Move ttMove_;      ///< Best move retrieved from the Transposition Table (searched first).
  Move countermove_; ///< Countermove heuristic response to the opponent's previous move.
  
  const std::array<std::array<Move, 2>, SearchConstants::MAX_PLY> *killers_; ///< Pointer to global killer heuristic table.
  const std::array<std::array<int, 64>, 64> *history_;                       ///< Pointer to global history heuristic table.
  const std::array<std::array<int, 64>, 16> *captureHist_;                   ///< Pointer to global capture history table.
  
  int ply_;          ///< Current distance from root.
  int threadId_;     ///< Thread ID for diversified move ordering in Lazy SMP.

  int nextCapture_ = 0;          ///< Index of the next capture to return.
  int nextKiller_ = 0;           ///< Index of the next killer to return.
  int nextQuiet_ = 0;            ///< Index of the next quiet move to return.
  
  std::array<Move, MAX_MOVES> &badCaptures_; ///< External buffer for captures that fail SEE.
  int badCaptureCount_ = 0;                  ///< Number of bad captures found.
  int nextBadCapture_ = 0;                   ///< Index of the next bad capture to return.
  
  bool capturesGenerated_ = false;           ///< True if capture moves have been scored and sorted.
  bool killersGenerated_ = false;            ///< True if killer moves have been extracted.
  bool quietsGenerated_ = false;             ///< True if quiet moves have been scored and sorted.
  
  int nextQueenPromo_ = 0;                   ///< Index of the next queen promotion to return.
  int queenPromoCount_ = 0;                  ///< Number of queen promotions found.
  
  bool countermoveReturned_ = false;         ///< True if the countermove has already been returned.
  int phase_ = 0;                            ///< Current phase: 0: Promos, 1: TT, 2: Captures, 3: Killers, 4: Countermove, 5: Quiets, 6: Bad Captures.

public:
  /**
   * @brief Constructs a MovePicker for a specific search node.
   * 
   * @param moves Generated legal moves.
   * @param ttMove Best move from TT.
   * @param killers Pointer to killer heuristics.
   * @param history Pointer to history heuristics.
   * @param captureHist Pointer to capture history heuristics.
   * @param captures Pre-allocated buffer for captures.
   * @param captureScores Pre-allocated buffer for capture scores.
   * @param quiets Pre-allocated buffer for quiets.
   * @param quietScores Pre-allocated buffer for quiet scores.
   * @param badCaptures Pre-allocated buffer for bad captures.
   * @param ply Current ply from root.
   * @param threadId Thread ID for diversity.
   * @param countermove Candidate countermove.
   */
  MovePicker(
      const Movelist &moves, Move ttMove,
      const std::array<std::array<Move, 2>, SearchConstants::MAX_PLY> *killers,
      const std::array<std::array<int, 64>, 64> *history,
      const std::array<std::array<int, 64>, 16> *captureHist,
      std::array<Move, MAX_MOVES> &captures,
      std::array<int, MAX_MOVES> &captureScores,
      std::array<Move, MAX_MOVES> &quiets,
      std::array<int, MAX_MOVES> &quietScores,
      std::array<Move, MAX_MOVES> &badCaptures,
      int ply, int threadId = 0, Move countermove = Move(0))
      : allMoves_(moves), captures_(captures), captureScores_(captureScores),
        quiets_(quiets), quietScores_(quietScores),
        badCaptures_(badCaptures), ttMove_(ttMove),
        countermove_(countermove),
        killers_(killers), history_(history), captureHist_(captureHist), 
        ply_(ply), threadId_(threadId) {}

  /**
   * @brief Returns the next best move according to the current phase.
   * 
   * Lazily processes and sorts moves only when a phase transition occurs.
   * 
   * @param board The current board state.
   * @param search Pointer to the active Negamax search instance for SEE evaluation.
   * @param inCheck True if the side to move is currently in check.
   * @return The next Move to evaluate, or Move(0) if all moves are exhausted.
   */
  Move nextMove(const Board &board, const Negamax *search, bool inCheck);

private:
  int scoreMove(const Move &move, const Board &board, const Negamax *search,
                bool isCapture);
  void generateAndScoreCaptures(const Board &board, const Negamax *search);
  void generateAndScoreKillers(const Board &board, const Negamax *search);
  void generateAndScoreQuiets(const Board &board, const Negamax *search);
  
  template<size_t N>
  Move selectBestMove(std::array<Move, N> &list, std::array<int, N> &scores,
                      int &nextIdx, int count);
};
