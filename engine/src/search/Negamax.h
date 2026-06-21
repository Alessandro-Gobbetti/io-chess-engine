#pragma once

/**
 * @file Negamax.h
 * @brief Alpha-Beta search implementation with enhancements.
 *
 * Implements the core search algorithm using the Negamax framework with 
 * Alpha-Beta pruning, Iterative Deepening, Transposition Tables, Late Move 
 * Reductions (LMR), Null Move Pruning, and various move ordering heuristics 
 * (Killer, History, Countermove).
 */

#include "../eval/IEvaluator.h"
#include "../eval/SimpleEvalContext.h"
#include "../tablebases/Tablebase.h"
#include "ISearch.h"
#include "TT.h"
#include "TimeManager.h"
#include <array>
#include <chrono>

#include "MovePicker.h"
#include "SearchHeuristics.h"

/**
 * @class Negamax
 * @brief Implementation of the main Alpha-Beta search algorithm.
 *
 * This class encapsulates the search state for a single thread, including
 * local transposition table statistics, move ordering heuristics, and time management.
 * It coordinates with other search threads through `SearchSharedData`.
 */
class Negamax : public ISearch {
  // Grant MovePicker access to heap-allocated heuristics
  friend class MovePicker;

private:
  IEvaluator &evalCtx_;              ///< Reference to the thread-local evaluator.
  TranspositionTable &tt_;           ///< Reference to the global transposition table.

  // LAZY SMP: Shared State
  std::shared_ptr<SearchSharedData> shared_; ///< Shared state for thread coordination.
  bool isMainThread_;                ///< True if this is the primary search thread.
  int threadId_;                     ///< Unique ID for this thread (used for move ordering diversity).

  // Thread-local statistics
  uint64_t localNodes_ = 0;          ///< Nodes searched by this thread.
  uint64_t localTbHits_ = 0;         ///< Tablebase hits by this thread.
  uint64_t localTtHits_ = 0;         ///< TT hits by this thread.
  int localSelDepth_ = 0;            ///< Maximum selective depth reached by this thread.
  static constexpr uint64_t NODE_FLUSH_INTERVAL = 1024; ///< How often to flush local stats to the shared global counters.

  // Time management
  TimeManager timeManager_;                  ///< Manages time limits and soft/hard stops.
  TimeManager::TimeAllocation timeAlloc_{};  ///< Pre-calculated time allocation constraints.
  float lastEvalCp_ = 0.0f;                  ///< Last evaluation score in centipawns (used for time decisions).

  // Iteration state tracking
  Move lastBestMove_;                ///< Best move found in the previous iterative deepening iteration.
  int lastScore_ = 0;                ///< Score returned in the previous iteration.
  int bestMoveChanges_ = 0;          ///< Count of how many times the best move changed at root.
  int scoreDrops_ = 0;               ///< Count of significant evaluation drops at root.
  std::array<uint64_t, 4096> rootNodeCounts_{}; ///< Number of nodes spent evaluating each root move.
  uint64_t searchNodes_ = 0;         ///< Total nodes searched during the current search session.

  std::unique_ptr<SearchHeuristics> h_; ///< Heap-allocated structure containing large arrays for search heuristics.

  /**
   * @brief Computes a hash key based on the pawn structure.
   * @param board The current board state.
   * @return The 64-bit Zobrist key for pawns only.
   */
  uint64_t getPawnKey(const Board& board) const;

  /**
   * @brief Computes a hash key based on non-pawn pieces of a given color.
   * @param board The current board state.
   * @param c The color of the pieces to hash.
   * @return The 64-bit non-pawn Zobrist key.
   */
  uint64_t getNonPawnKey(const Board& board, Color c) const;

  // Search context tracking arrays
  std::array<Move, SearchConstants::MAX_PLY> prevMove_;  ///< Records the move made at each ply to reach the current state.
  std::array<Piece, SearchConstants::MAX_PLY> prevPiece_{}; ///< Records the piece that moved at each ply.
  std::array<int, SearchConstants::MAX_PLY> pvLength_;   ///< Tracks the length of the principal variation at each ply.
  std::array<int, SearchConstants::MAX_PLY> evalHistory_{}; ///< Tracks static evaluation history for pruning thresholds.
  
  int currentIter_ = 0; ///< The current depth iteration in Iterative Deepening.

  std::vector<Move> searchMoves_; ///< Restricts the search to only these root moves (used for 'searchmoves' command).

  InfoCallback infoCallback_; ///< Callback function to output search progress to the UCI protocol.

  SimpleEvalContext simpleEval_; ///< A fast, simple evaluator used for pre-NN filtering or lazy evaluation.

public:
  /**
   * @brief Constructs a new Negamax search instance.
   * 
   * @param eval Thread-local evaluator instance.
   * @param table Global transposition table.
   * @param shared Shared state for Lazy SMP coordination.
   * @param isMainThread True if this instance is the primary search thread.
   * @param threadId Unique identifier for this thread.
   */
  Negamax(IEvaluator &eval, TranspositionTable &table,
          std::shared_ptr<SearchSharedData> shared, bool isMainThread,
          int threadId = 0);

  Move startSearch(Board &root, const SearchParams &params) override;
  
  void stop() override {
    if (shared_)
      shared_->stop = true;
  }
  
  bool isSearching() const override { return shared_ && !shared_->stop; }
  
  void setInfoCallback(InfoCallback callback) override {
    infoCallback_ = callback;
  }
  
  uint64_t getNodes() const override {
    return shared_ ? shared_->totalNodes.load(std::memory_order_relaxed) : 0;
  }

private:
  /**
   * @brief The core recursive Alpha-Beta search function.
   * 
   * @param board The current board state.
   * @param depth The remaining search depth.
   * @param alpha The lower bound of the current search window.
   * @param beta The upper bound of the current search window.
   * @param ply The current distance from the root.
   * @param allowNull True if Null Move Pruning is permitted at this node.
   * @param extensions The number of depth extensions applied so far along this path.
   * @param prevWasCapture True if the move that led to this node was a capture.
   * @param cutnode True if we expect this node to fail high (used for LMR scaling).
   * @param excludedMove A move to ignore during search (used for Singular Extensions).
   * @return The backed-up minimax score for the node.
   */
  int alphaBeta(Board &board, int depth, int alpha, int beta, int ply,
                bool allowNull, int extensions, bool prevWasCapture = false, bool cutnode = false, Move excludedMove = Move(0));

  /**
   * @brief Quiescence search to resolve tactical sequences before static evaluation.
   * 
   * Searches only captures and promotions to ensure the evaluation is stable.
   * 
   * @param board The current board state.
   * @param alpha The lower bound.
   * @param beta The upper bound.
   * @param ply The distance from the root.
   * @return The resolved stable score.
   */
  int quiescence(Board &board, int alpha, int beta, int ply);

  /**
   * @brief Determines if the position can bypass a full neural network evaluation.
   * 
   * @param board The board state.
   * @param depth Remaining depth.
   * @param alpha Alpha bound.
   * @param beta Beta bound.
   * @param ply Distance from root.
   * @param fastScore Score from the fast/simple evaluator.
   * @param outBound Output bound (Upper or Lower) if skipping is permitted.
   * @param prevWasCapture True if previous move was a capture.
   * @param pvNode True if this is a Principal Variation node.
   * @return True if the evaluation can be skipped.
   */
  bool canLazySkip(const Board &board, int depth, int alpha, int beta, int ply,
                   int fastScore, Bound &outBound, bool prevWasCapture,
                   bool pvNode);

  /**
   * @brief Scores a move rapidly for sorting during move generation.
   * 
   * @param move The move to score.
   * @param board The current board state.
   * @param ttMove The best move retrieved from the Transposition Table.
   * @param ply The current distance from the root.
   * @param isCapture True if the move is a capture.
   * @return The heuristic sorting score.
   */
  int scoreMoveFast(const Move &move, const Board &board, Move ttMove, int ply,
                    bool isCapture);

  /**
   * @brief Updates killer move heuristics when a quiet move causes a beta cutoff.
   * 
   * @param move The move that caused the cutoff.
   * @param ply The distance from root where the cutoff occurred.
   */
  void updateKillers(Move move, int ply);

  /**
   * @brief Updates history and continuation history heuristics based on search results.
   * 
   * @param board The board state before the move.
   * @param move The move executed.
   * @param bonus The bonus/penalty to apply (based on search depth).
   * @param ply The current distance from root.
   */
  void updateHistory(const Board& board, Move move, int bonus, int ply);

  friend class MovePicker; ///< MovePicker requires access to `scoreMoveFast` and heuristics.

  /**
   * @brief Static Exchange Evaluation (SEE) for a move.
   * 
   * Evaluates the material consequence of a sequence of captures on a single square.
   * 
   * @param board The board state.
   * @param move The capture move to evaluate.
   * @return The material gain/loss score.
   */
  int see(const Board &board, Move move) const;

  /**
   * @brief Returns the base material value of a piece type for SEE calculations.
   */
  int pieceValue(PieceType pt) const;

  /**
   * @brief Applies historical corrections to the raw neural network evaluation.
   * 
   * Corrects systematic evaluation errors based on pawn structures and piece configurations.
   * 
   * @param rawEval The raw static evaluation.
   * @param board The board state.
   * @param ply The distance from root.
   * @return The history-corrected evaluation.
   */
  int applyCorrHist(int rawEval, const Board &board, int ply) const;

  // --- Extension Condition Checkers ---
  
  /**
   * @brief Determines if the pawn on the given square is a passed pawn.
   * @param board The current board state.
   * @param sq The square of the pawn.
   * @param side The color of the pawn.
   * @return True if the pawn has no opposing pawns blocking its path to promotion.
   */
  bool isPassedPawn(const Board &board, Square sq, Color side) const;

  /**
   * @brief Determines if the move pushes a pawn to the 7th rank (or 2nd for black).
   * @param board The current board state.
   * @param move The move to evaluate.
   * @return True if it is a 7th rank pawn push.
   */
  bool isPawnPush7th(const Board &board, Move move) const;

  // --- Time Management Helpers ---

  /**
   * @brief Syncs the thread-local elapsed time to the shared data for the main thread.
   */
  void syncTimeToShared();

  /**
   * @brief Checks if the search should be aborted due to time limits or external stop signals.
   * @return True if the search must stop immediately.
   */
  bool shouldStop() const;

  /**
   * @brief Calculates the elapsed time since the search started.
   * @return Elapsed time in milliseconds.
   */
  int64_t elapsedMs() const;

  /**
   * @brief Decides whether to stop the search early at the end of an Iterative Deepening iteration.
   * 
   * Triggers soft time limits early if the best move seems stable and no score drops have occurred.
   * 
   * @param depth The completed search depth.
   * @param score The best score achieved at this depth.
   * @param bestMove The best move found.
   * @return True if the search should be terminated.
   */
  bool shouldStopIteration(int depth, int score, Move bestMove) const;

  // --- General Utilities ---

  /**
   * @brief Returns true if the move is a capture.
   */
  bool isCapture(const Board &board, Move move) const;

  /**
   * @brief Returns true if the position is a theoretical draw (e.g. 50-move rule, 3-fold repetition, insufficient material).
   */
  bool isDraw(const Board &board) const;

  /**
   * @brief Returns the mate score adjusted for the current ply (so closer mates score higher).
   */
  int mateScore(int ply) const;

  /**
   * @brief Checks if the given score is within the checkmate score range.
   */
  bool isMateScore(int score) const {
    return std::abs(score) >= SearchConstants::MATE_IN_MAX - SearchConstants::MAX_PLY;
  }

  /**
   * @brief Resets transient state arrays and variables before starting a new search.
   */
  void clearState();

  /**
   * @brief Flushes local statistics (nodes, hits) to the global shared atomic counters.
   */
  void flushLocalSearchStats();
};