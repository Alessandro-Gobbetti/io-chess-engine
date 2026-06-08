#pragma once

/**
 * @file ISearch.h
 * @brief Search interface and shared data structures.
 *
 * Defines the common interface for search algorithms (e.g., Negamax and MCTS)
 * and the shared state structures used for Lazy SMP, time management, and tree export.
 */

#include "../Types.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

/**
 * @struct SearchSharedData
 * @brief Thread-safe shared state for search coordination and Lazy SMP.
 *
 * This struct holds the atomic variables used to coordinate multiple search
 * threads, manage time limits, and collect statistics across the entire engine.
 */
struct SearchSharedData {
  std::atomic<bool> stop{false};            ///< Global flag to immediately stop all search threads.
  std::atomic<bool> disableTablebase{false}; ///< Disables tablebase probing (e.g., when Syzygy coverage is incomplete).
  std::atomic<uint64_t> totalNodes{0};       ///< Global node count across all threads.
  std::atomic<int> selDepth{0};              ///< Maximum selective depth reached by any thread.
  std::atomic<uint64_t> tbHits{0};           ///< Total tablebase hits across all threads.
  std::atomic<uint64_t> ttHits{0};           ///< Total transposition table hits across all threads.

  // =========================================================================
  // LAZY SMP: Shared state for thread coordination
  // =========================================================================
  
  std::atomic<int> mainThreadScore{0}; ///< Main thread's current score (used by helpers for aspiration windows).
  std::atomic<int> mainThreadDepth{0}; ///< Main thread's completed depth (used by helpers for early termination).
  std::atomic<int> targetDepth{0};     ///< Target search depth (set by main thread at search start).
  int numThreads{1};                   ///< Total number of active search threads.

  // Time management (written by Main, read by Helpers)
  std::atomic<int64_t> startTimeNs{0}; ///< Search start time in steady_clock nanoseconds (atomic for ponderhit updates).
  int64_t softLimit = 0;               ///< Soft time limit in milliseconds.
  int64_t hardLimit = 0;               ///< Hard time limit in milliseconds.

  /**
   * @struct SearchConfig
   * @brief Runtime tunable parameters for the search algorithm.
   */
  struct SearchConfig {
    int razorMarginD1 = 350;             ///< Razoring margin at depth 1.
    int razorMarginD2 = 700;             ///< Razoring margin at depth 2.
    int razorMarginD3 = 900;             ///< Razoring margin at depth 3.
    int futilityMargin = 200;            ///< Futility pruning margin.
    int reverseFutilityMargin = 120;     ///< Reverse futility (static null move) margin.
    int lmpBase = 3;                     ///< Late Move Pruning base threshold.
    int lmrThreshold = 4;                ///< Late Move Reduction threshold.
    int nullMoveR = 3;                   ///< Null Move Pruning depth reduction factor.
    int deltaMargin = 200;               ///< Delta pruning margin.
    int singularMargin = 50;             ///< Singular extension margin.
    int singularMinDepth = 6;            ///< Minimum depth required for singular extensions.
    int internalPruningMargin = 150;     ///< Margin for internal node pruning.
    int corrWeight = 10;                 ///< Weight applied to the correction history heuristic.
    bool enableCorrHist = true;          ///< Flag to enable or disable the correction history heuristic.
  } config;

  // Lazy Eval Tuning
  int lazyEvalMaxDepth = 6;      ///< Maximum depth to apply lazy evaluation.
  int lazyEvalBaseMargin = 350;  ///< Base margin for lazy evaluation (approx 1 Sigma of residual error).
  int lazyEvalDepthMargin = 60;  ///< Depth multiplier margin for lazy evaluation.
  int lazyEvalMinMargin = 200;   ///< Absolute minimum margin for lazy evaluation.
  bool enableLazyEval = false;   ///< Flag to enable or disable lazy evaluation.

  /**
   * @enum PruneReason
   * @brief Enumeration of reasons why a node might be pruned (used for tree visualization).
   */
  enum class PruneReason : uint8_t {
    NONE = 0,           ///< Not pruned - fully searched
    LMP,                ///< Late Move Pruning
    FUTILITY,           ///< Futility Pruning
    SEE,                ///< Static Exchange Evaluation pruning
    DRAW,               ///< Draw by repetition or 50-move rule
    RAZOR,              ///< Razoring (drop to qsearch)
    REVERSE_FUTILITY,   ///< Reverse Futility / Static Null Move
    NULL_MOVE,          ///< Null Move Pruning cutoff
    INTERNAL,           ///< Internal Node Pruning (Eval Margin)
    LAZY_EVAL,          ///< Lazy Eval skip (fast score sufficient)
    TB_CUTOFF,          ///< Tablebase cutoff
    TT_CUTOFF,          ///< Transposition table cutoff
    BETA_CUTOFF,        ///< Normal beta cutoff
    DEPTH_LIMIT,        ///< Beyond exportTreeDepth and not PV
    SINGULAR_SKIP       ///< Skipped during singular extension search
  };

  /**
   * @struct VizNode
   * @brief Represents a single node in the search tree for export and visualization.
   */
  struct VizNode {
    uint64_t parentHash;       ///< Zobrist hash of the parent position.
    uint64_t childHash;        ///< Zobrist hash of the child position.
    std::string move;          ///< UCI string representation of the move.
    int staticEval;            ///< NN evaluation at node (before search).
    int searchScore;           ///< Minimax backed-up score (after search).
    int depth;                 ///< Search depth remaining.
    int ply;                   ///< Ply from root (0 = root).
    int moveOrder;             ///< Order this move was tried (0 = first, 1 = second, etc.).
    uint64_t subtreeNodes;     ///< Nodes searched below this move.
    bool isPV;                 ///< True if on principal variation.
    PruneReason pruneReason;   ///< Why this node was pruned.
  };
  
  std::vector<VizNode> vizTree; ///< Stores edges for top layers of the search tree.
  bool exportTree = false;      ///< Flag to enable tree export visualization.
  int exportTreeDepth = 4;      ///< Maximum depth to record nodes for tree visualization.
};

/**
 * @struct SearchParams
 * @brief Parameters defining the constraints for a search operation.
 */
struct SearchParams {
  int wtime = 0;     ///< White time remaining in ms.
  int btime = 0;     ///< Black time remaining in ms.
  int winc = 0;      ///< White increment per move in ms.
  int binc = 0;      ///< Black increment per move in ms.
  int movestogo = 0; ///< Moves until next time control.
  int depth = SearchConstants::DEFAULT_DEPTH; ///< Target depth limit.
  int nodes = 0;     ///< Node limit (0 = unlimited).
  int movetime = 0;  ///< Fixed time per move in ms.
  bool infinite = false; ///< True if search should run until explicitly stopped.
  bool ponder = false;   ///< True if thinking during the opponent's time (wait for ponderhit/stop).
  int multiPV = 1;       ///< Number of principal variations to search (1 = normal, >1 = analysis mode).

  std::vector<Move> searchMoves; ///< Specific root moves to search (if empty, search all legal moves).
};

/**
 * @typedef InfoCallback
 * @brief Callback function type for sending UCI 'info' updates during search.
 */
using InfoCallback = std::function<void(int depth, int score, int nodes,
                                        int nps, const std::vector<Move> &pv)>;

/**
 * @class ISearch
 * @brief Abstract interface for search algorithms.
 *
 * This interface defines the standard lifecycle of a search algorithm, allowing
 * the engine to switch between implementations (e.g., Negamax vs. MCTS).
 */
class ISearch {
public:
  virtual ~ISearch() = default;

  /**
   * @brief Starts the search process on the given root board.
   * 
   * This is a blocking call that will execute the search algorithm until the
   * stopping conditions in `params` (or `stop()`) are met.
   * 
   * @param root The starting board position.
   * @param params The time and depth constraints for the search.
   * @return The best move found.
   */
  virtual Move startSearch(Board &root, const SearchParams &params) = 0;

  /**
   * @brief Asynchronously signals the search to stop immediately.
   * 
   * This method must be thread-safe.
   */
  virtual void stop() = 0;

  /**
   * @brief Checks if the search algorithm is currently running.
   * 
   * @return True if a search is in progress.
   */
  virtual bool isSearching() const = 0;

  /**
   * @brief Sets the callback function for periodic information updates.
   * 
   * @param callback The function to call with depth, score, node count, NPS, and PV.
   */
  virtual void setInfoCallback(InfoCallback callback) = 0;

  /**
   * @brief Retrieves the total number of nodes evaluated by the search algorithm.
   * 
   * @return The node count.
   */
  virtual uint64_t getNodes() const = 0;
};