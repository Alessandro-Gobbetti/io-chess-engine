#pragma once
// UCI.h - Universal Chess Interface protocol implementation
// Handles communication with claim GUIs

#include "../Types.h"
#include "../book/polyglot_book.hpp"

#include "../eval/Evaluator.h"
#include "../search/ISearch.h"
#include "../search/MCTS.h"
#include "../search/Negamax.h"
#include "../search/TimeManager.h"
#include "../search/TT.h"
#include "../tablebases/Tablebase.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

// Engine constant for versioning
static constexpr const char *ENGINE_VERSION = "1.0";

class UciProtocol {
private:
  // Engine name and author
  static constexpr const char *ENGINE_NAME = "io-chess-engine";
  static constexpr const char *ENGINE_AUTHOR = "Alessandro Gobbetti";

  // Persistent Resources (loaded once)
  std::vector<std::unique_ptr<Evaluator>> evaluators_;
  std::vector<std::unique_ptr<IEvaluator>> evalCtxs_;
  TranspositionTable tt_;
  std::unique_ptr<PolyglotBook> book_;

  // Threading and Search State
  std::vector<std::thread> threadPool_;
  std::shared_ptr<SearchSharedData> searchData_;
  int numThreads_ = 1;
  int evalThreads_ = 0; // 0 = Auto

  // The active search algorithm
  std::unique_ptr<ISearch> searcher_;
  std::thread searchThread_;
  std::atomic<bool> searchRunning_{false};
  std::atomic<bool> pondering_{false};
  std::atomic<bool> ponderResultReady_{false};
  Move ponderResultMove_{};
  Move ponderMove_{}; // Expected opponent reply (2nd move in PV)
  SearchParams ponderParams_{}; // Time params from go ponder for ponderhit use
  std::mutex ponderMutex_;

  // Current position
  Board currentBoard_;

  // Options
  std::string modelPath_;
  std::string tbPath_;    // Tablebase path
  std::string bookPath_;  // Opening book path
  std::string bookPath2_; // Secondary opening book path
  std::unique_ptr<PolyglotBook> book2_;
  int consecutiveBookMisses_ = 0;
  static constexpr int MAX_BOOK_MISSES = 3;
  size_t hashSizeMB_ = 128;
  bool useMCTS_ = false;
  bool useSimpleEval_ = false; // If true, use simple eval; if false, use NN
  bool useBook_ = true;
  bool analyseMode_ = false; // If true, probe book but don't halt search
  bool useGPU_ = false; // Simplified from ExecutionProvider
  bool chess960_ = false; // Chess960/FRC mode
  float aggression_ =
      0.0f; // WDL conversion aggression: -1.0 (defensive) to +1.0 (aggressive)
  int lazyEvalMaxDepth_ = 6;
  int lazyEvalBaseMargin_ = 500;
  int lazyEvalDepthMargin_ = 100;
  bool enableLazyEval_ = true;
  int evalScaleBase_ = 750;
  int evalScaleWeight_ = 25;
  bool enableEvalNormalization_ = true;
  SearchSharedData::SearchConfig searchConfig_;
  
  // Tree Export Options
  bool exportTree_ = false;
  int exportTreeDepth_ = 4;

public:
  explicit UciProtocol(const std::string &modelPath, bool useSimpleEval = false,
                       const std::string &tbPath = "",
                       const std::string &bookPath = "",
                       const std::string &bookPath2 = "", int evalThreads = 0);
  ~UciProtocol();

  // Main command loop (blocking)
  void loop();

private:
  // UCI command handlers
  void handleUci();
  void handleIsReady();
  void handleSetOption(std::istringstream &is);
  void handleUciNewGame();
  void handlePosition(std::istringstream &is);
  void handleGo(std::istringstream &is);
  void handlePonderHit();
  void handleStop();
  void handleQuit();
  void handleEval();

  // Helper functions
  void initializeEngine();
  static std::string formatScore(int score);
  void sendInfo(int depth, int score, int nodes, int nps,
                const std::vector<Move> &pv, const Board &board);
  void sendBestMove(Move move, Move ponderMove = Move());
};