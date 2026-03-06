  #pragma once
// Negamax.h - Alpha-Beta search with enhancements
// Implements: Iterative Deepening, TT cutoffs, LMR, Null Move, Killer/History

#include "../eval/IEvaluator.h"
#include "../eval/SimpleEvalContext.h"
#include "../tablebases/Tablebase.h"
#include "ISearch.h"
#include "TT.h"
#include "TimeManager.h"
#include <array>
#include <chrono>

// Forward declarations
class Negamax;

// MovePicker: Lazily selects the best move at each step
// Avoids sorting all moves upfront - crucial for cutoff-heavy nodes
class MovePicker {
private:
  Movelist allMoves_;          // All legal moves (generated once)
  
  // === OPTIMIZATION: Fixed-size arrays to avoid dynamic allocations ===
  // Max legal moves in any chess position is 218, but typical positions have <50
  static constexpr int MAX_MOVES = 256;
  std::array<Move, MAX_MOVES> captures_;
  std::array<int, MAX_MOVES> captureScores_;
  int captureCount_ = 0;
  
  std::array<Move, 2> killerMoves_; // Killer moves (max 2)
  int killerCount_ = 0;
  
  std::array<Move, MAX_MOVES> quiets_;
  std::array<int, MAX_MOVES> quietScores_;
  int quietCount_ = 0;
  
  Move ttMove_;      // Best move from TT (searched first)
  Move countermove_; // Countermove for opponent's previous move
  const std::array<std::array<Move, 2>, SearchConstants::MAX_PLY> *killers_;
  const std::array<std::array<int, 64>, 64> *history_;
  const std::array<std::array<int, 64>, 16> *captureHist_;
  int ply_;
  int threadId_; // Thread ID for diversified move ordering

  int nextCapture_ = 0;
  int nextKiller_ = 0;
  int nextQuiet_ = 0;
  std::array<Move, MAX_MOVES> badCaptures_;
  int badCaptureCount_ = 0;
  int nextBadCapture_ = 0;
  bool capturesGenerated_ = false;
  bool killersGenerated_ = false;
  bool quietsGenerated_ = false;
  int nextQueenPromo_ = 0;
  int queenPromoCount_ = 0;
  bool countermoveReturned_ = false;
  int phase_ = 0; // 0: Queen promos, 1: TT move, 2: captures, 3: killers, 4:
                  // countermove, 5: quiets, 6: bad captures

public:
  MovePicker(
      const Movelist &moves, Move ttMove,
      const std::array<std::array<Move, 2>, SearchConstants::MAX_PLY> *killers,
      const std::array<std::array<int, 64>, 64> *history,
      const std::array<std::array<int, 64>, 16> *captureHist,
      int ply, int threadId = 0, Move countermove = Move(0))
      : allMoves_(moves), ttMove_(ttMove), countermove_(countermove),
        killers_(killers), history_(history), captureHist_(captureHist), 
        ply_(ply), threadId_(threadId) {}

  // Get next best move
  Move nextMove(const Board &board, const Negamax *search, bool inCheck);

private:
  int scoreMove(const Move &move, const Board &board, const Negamax *search,
                bool isCapture);
  void generateAndScoreCaptures(const Board &board, const Negamax *search);
  void generateAndScoreKillers(const Board &board, const Negamax *search);
  void generateAndScoreQuiets(const Board &board, const Negamax *search);
  
  // Template for fixed-size array selection (avoids heap allocations)
  template<size_t N>
  Move selectBestMove(std::array<Move, N> &list, std::array<int, N> &scores,
                      int &nextIdx, int count);
};

// SearchHeuristics: Large search data structures consolidated for heap allocation
struct SearchHeuristics {
  std::array<std::array<Move, 2>, SearchConstants::MAX_PLY> killers_{};
  std::array<std::array<int, 64>, 64> history_{};
  std::array<std::array<Move, 64>, 64> countermove_{};
  std::array<std::array<std::array<std::array<int, 64>, 16>, 64>, 16> contHist_{};
  std::array<std::array<int, 64>, 16> captureHist_{};
  std::array<std::array<int16_t, 16384>, 2> pawnCorrHist_{};
  std::array<std::array<std::array<int16_t, 16384>, 2>, 2> nonPawnCorrHist_{};
  std::array<std::array<std::array<std::array<int16_t, 64>, 16>, 64>, 16> contCorrHist_{};
  std::array<std::array<Move, SearchConstants::MAX_PLY>, SearchConstants::MAX_PLY> pvTable_{};
  std::array<int, SearchConstants::MAX_PLY> failHighCount_{};
};

class Negamax : public ISearch {
  // Grant MovePicker access to heap-allocated heuristics
  friend class MovePicker;

private:
  // Search context
  IEvaluator &evalCtx_;
  TranspositionTable &tt_;

  // LAZY SMP: Shared State
  std::shared_ptr<SearchSharedData> shared_;
  bool isMainThread_;
  int threadId_; // Thread ID for diversified move ordering

  // Thread-local node counter (flush to shared periodically to avoid
  // contention)
  uint64_t localNodes_ = 0;
  static constexpr uint64_t NODE_FLUSH_INTERVAL = 1024;

  // Time management
  TimeManager timeManager_;
  TimeManager::TimeAllocation timeAlloc_{};
  float lastEvalCp_ = 0.0f; // Last search eval for time decisions

  // Time management state (for dynamic adjustment)
  Move lastBestMove_;       // Best move from previous iteration
  int lastScore_ = 0;       // Score from previous iteration
  int bestMoveChanges_ = 0; // How many times best move changed
  int scoreDrops_ = 0;      // How many times score dropped significantly
  std::array<uint64_t, 4096> rootNodeCounts_{}; // Nodes spent on each root move
  uint64_t searchNodes_ = 0; // Total nodes this search

  // Heap-allocated heuristics to prevent stack overflow
  std::unique_ptr<SearchHeuristics> h_;

  // Zobrist helper functions
  uint64_t getPawnKey(const Board& board) const;
  uint64_t getNonPawnKey(const Board& board, Color c) const;

  // Previous move tracking for recapture extension and countermove
  std::array<Move, SearchConstants::MAX_PLY>
      prevMove_; // [ply] -> move made at that ply
  std::array<Piece, SearchConstants::MAX_PLY>
      prevPiece_{}; // [ply] -> piece that moved

  std::array<int, SearchConstants::MAX_PLY> pvLength_;

  // State-Aware LMR and SE Tracking
  std::array<int, SearchConstants::MAX_PLY> evalHistory_{};
  int currentIter_ = 0;

  // Root move filtering (for searchMoves and MultiPV)
  std::vector<Move> searchMoves_;

  // Info callback
  InfoCallback infoCallback_;

  // Cheap evaluator for pre-NN filtering
  SimpleEvalContext simpleEval_;

public:
  Negamax(IEvaluator &eval, TranspositionTable &table,
          std::shared_ptr<SearchSharedData> shared, bool isMainThread,
          int threadId = 0);

  // ISearch interface
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
  // Core search functions
  int alphaBeta(Board &board, int depth, int alpha, int beta, int ply,
                bool allowNull, int extensions, bool prevWasCapture = false, bool cutnode = false, Move excludedMove = Move(0));
  int quiescence(Board &board, int alpha, int beta, int ply);

  // Helper for Lazy Eval Pruning
  bool canLazySkip(const Board &board, int depth, int alpha, int beta, int ply,
                   int fastScore, Bound &outBound, bool prevWasCapture,
                   bool pvNode);

  // Move ordering (now used by MovePicker)
  int scoreMoveFast(const Move &move, const Board &board, Move ttMove, int ply,
                    bool isCapture);
  void updateKillers(Move move, int ply);
  void updateHistory(const Board& board, Move move, int bonus, int ply);

  // Helper for MovePicker to compute move scores
  friend class MovePicker;

  // Static Exchange Evaluation (SEE)
  int see(const Board &board, Move move) const;
  int pieceValue(PieceType pt) const;

  // Extension helpers
  bool isPassedPawn(const Board &board, Square sq, Color side) const;
  bool isPawnPush7th(const Board &board, Move move) const;

  // Time management (delegated to TimeManager)
  void syncTimeToShared();
  bool shouldStop() const;
  int64_t elapsedMs() const;
  bool shouldStopIteration(int depth, int score, Move bestMove) const;

  // Utility
  bool isCapture(const Board &board, Move move) const;
  bool isDraw(const Board &board) const;
  int mateScore(int ply) const;
  bool isMateScore(int score) const {
    return std::abs(score) >=
           SearchConstants::MATE_IN_MAX - SearchConstants::MAX_PLY;
  }

  // Reset state for new search
  void clearState();
};