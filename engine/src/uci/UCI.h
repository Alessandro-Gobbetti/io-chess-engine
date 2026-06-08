#pragma once

/**
 * @file UCI.h
 * @brief Universal Chess Interface protocol implementation.
 *
 * Handles communication between the engine and external GUIs (e.g., Cutechess).
 * Parses standard UCI commands and drives the underlying search and evaluation components.
 */

#include "UciOptions.h"

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

/**
 * @class UciProtocol
 * @brief Main engine controller implementing the UCI protocol.
 *
 * Orchestrates parsing input from standard input, managing engine options,
 * starting/stopping search threads, and reporting results to standard output.
 */
class UciProtocol {
private:
  // Engine name and author
  static constexpr const char *ENGINE_NAME = "io-chess-engine";
  static constexpr const char *ENGINE_AUTHOR = "Alessandro Gobbetti!";

  // Persistent Resources (loaded once)
  std::vector<std::unique_ptr<Evaluator>> evaluators_;
  std::vector<std::unique_ptr<IEvaluator>> evalCtxs_;
  TranspositionTable tt_;
  std::unique_ptr<PolyglotBook> book_;

  // Threading and Search State
  std::vector<std::thread> threadPool_;
  std::shared_ptr<SearchSharedData> searchData_;

  // The active search algorithm
  std::unique_ptr<ISearch> searcher_;
  std::thread searchThread_;
  std::atomic<bool> searchRunning_{false};
  std::atomic<bool> searchMayRunForever_{false};
  std::atomic<bool> pondering_{false};
  std::atomic<bool> ponderResultReady_{false};
  Move ponderResultMove_{};
  Move ponderMove_{}; ///< Expected opponent reply (2nd move in PV)
  SearchParams ponderParams_{}; ///< Time params from go ponder for ponderhit use
  std::mutex ponderMutex_;

  // Current position
  Board currentBoard_;

  // Options
  UciOptions options_;
  std::unique_ptr<PolyglotBook> book2_;
  int consecutiveBookMisses_ = 0;
  static constexpr int MAX_BOOK_MISSES = 3;

  std::string pendingTTLoadFile_;
  uint64_t ttSnapshotCounter_ = 0;

  // Per-search metric baselines (used to report deltas in info lines)
  uint64_t searchBaseFullRebuilds_ = 0;
  uint64_t searchBaseTbHits_ = 0;
  uint64_t searchBaseTtHits_ = 0;

public:
  /**
   * @brief Constructs the UCI Protocol handler.
   */
  explicit UciProtocol(const std::string &modelPath, bool useSimpleEval = false,
                       const std::string &tbPath = "",
                       const std::string &bookPath = "",
                       const std::string &bookPath2 = "", int evalThreads = 0);
  ~UciProtocol();

  /**
   * @brief The main blocking loop that reads commands from standard input.
   */
  void loop();

  // Callbacks for options
  void resizeHash(size_t mb);
  void updateThreads(int numThreads);
  void reinitEngine();
  void updateEvalContexts();
  void initTablebase();
  void loadBooks();
  void setTTDisabled(bool disable);
  void setPendingTTLoadFile(const std::string& file) { pendingTTLoadFile_ = file; }

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
  void sendBestMove(Move move, Move ponderMove = Move(Move::NO_MOVE));
  void saveTTSnapshotForMove(const std::string &moveUci);
  static std::string sanitizeForFilename(const std::string &value);
};