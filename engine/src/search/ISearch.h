#pragma once
// ISearch.h - Search interface
// Defines common interface for Negamax and MCTS searchers

#include "../Types.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

struct SearchSharedData {
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> totalNodes{0};

  // =========================================================================
  // LAZY SMP: Shared state for thread coordination
  // =========================================================================
  
  // Main thread's current score (for helper aspiration windows)
  // Helpers use this to set tighter search windows, reducing tree size
  std::atomic<int> mainThreadScore{0};
  
  // Main thread's completed depth (for helper early termination)
  // Helpers stop when main finishes target depth to avoid wasted work
  std::atomic<int> mainThreadDepth{0};
  
  // Target search depth (set by main thread at search start)
  std::atomic<int> targetDepth{0};
  
  // Number of search threads (for diversity calculations)
  int numThreads{1};

  // Time management (written by Main, read by Helpers)
  std::chrono::steady_clock::time_point startTime;
  int64_t softLimit = 0;
  int64_t hardLimit = 0;
  // Runtime Tunable Parameters
  struct SearchConfig {
    int razorMarginD1 = 350;
    int razorMarginD2 = 700;
    int razorMarginD3 = 900;
    int futilityMargin = 200;
    int reverseFutilityMargin = 120;
    int lmpBase = 3;
    int lmrThreshold = 4;
    int nullMoveR = 3;
    int deltaMargin = 200;
    int singularMargin = 50;
    int singularMinDepth = 6;
    int internalPruningMargin = 150;
    int corrWeight = 10;
    bool enableCorrHist = true;
  } config;

  // Lazy Eval Tuning
  int lazyEvalMaxDepth = 6;
  int lazyEvalBaseMargin =
      350; // Conservative margin (approx 1 Sigma of residual error)
  int lazyEvalDepthMargin = 60;
  int lazyEvalMinMargin = 200;
  bool enableLazyEval = false;

  // Search Tree Visualization
  // Pruning reasons for tree visualization
  enum class PruneReason : uint8_t {
    NONE = 0,           // Not pruned - fully searched
    // Move-level pruning (inside move loop)
    LMP,                // Late Move Pruning
    FUTILITY,           // Futility Pruning
    SEE,                // Static Exchange Evaluation pruning
    // Node-level pruning (before move generation)
    DRAW,               // Draw by repetition or 50-move rule
    RAZOR,              // Razoring (drop to qsearch)
    REVERSE_FUTILITY,   // Reverse Futility / Static Null Move
    NULL_MOVE,          // Null Move Pruning cutoff
    INTERNAL,           // Internal Node Pruning (Eval Margin)
    LAZY_EVAL,          // Lazy Eval skip (fast score sufficient)
    TB_CUTOFF,          // Tablebase cutoff
    TT_CUTOFF,          // Transposition table cutoff
    // Result indicators
    BETA_CUTOFF,        // Normal beta cutoff
    DEPTH_LIMIT,        // Beyond exportTreeDepth and not PV
    SINGULAR_SKIP       // Skipped during singular extension search
  };

  struct VizNode {
    uint64_t parentHash;
    uint64_t childHash;
    std::string move;        // UCI string
    int staticEval;          // NN evaluation at node (before search)
    int searchScore;         // Minimax backed-up score (after search)
    int depth;               // Search depth remaining
    int ply;                 // Ply from root (0 = root)
    int moveOrder;           // Order this move was tried (0 = first, 1 = second, etc.)
    uint64_t subtreeNodes;   // Nodes searched below this move
    bool isPV;               // True if on principal variation
    PruneReason pruneReason; // Why this node was pruned
  };
  std::vector<VizNode> vizTree; // Stores edges for top layers
  bool exportTree = false;
  int exportTreeDepth = 4;
};

struct SearchParams {
  int wtime = 0;     // White time in ms
  int btime = 0;     // Black time in ms
  int winc = 0;      // White increment in ms
  int binc = 0;      // Black increment in ms
  int movestogo = 0; // Moves until next time control
  int depth = SearchConstants::DEFAULT_DEPTH;
  int nodes = 0;    // Node limit (0 = unlimited)
  int movetime = 0; // Fixed time per move in ms
  bool infinite = false;
  bool ponder = false; // If true, think until ponderhit/stop
  int multiPV = 1; // Number of principal variations to search (1 = normal, >1 =
                   // analysis mode)

  // Search specific moves only
  std::vector<Move> searchMoves;
};

// Callback for UCI info updates
using InfoCallback = std::function<void(int depth, int score, int nodes,
                                        int nps, const std::vector<Move> &pv)>;

// Abstract Interface for search algorithms
class ISearch {
public:
  virtual ~ISearch() = default;

  // Blocking call that runs the search logic and returns best move
  virtual Move startSearch(Board &root, const SearchParams &params) = 0;

  // Async signal to stop immediately (must be thread-safe)
  virtual void stop() = 0;

  // Check if search is running
  virtual bool isSearching() const = 0;

  // Set callback for info output
  virtual void setInfoCallback(InfoCallback callback) = 0;

  // Get nodes searched
  virtual uint64_t getNodes() const = 0;
};