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

  // --- Callbacks for options ---
  
  /** @brief Resizes the transposition table memory block. */
  void resizeHash(size_t mb);
  
  /** @brief Restarts the search thread pool with the specified number of threads. */
  void updateThreads(int numThreads);
  
  /** @brief Performs a full re-initialization of the engine and evaluator state. */
  void reinitEngine();
  
  /** @brief Refreshes the thread-local evaluation contexts. */
  void updateEvalContexts();
  
  /** @brief Reloads the Syzygy tablebases from the configured path. */
  void initTablebase();
  
  /** @brief Reloads the Polyglot opening books from the configured paths. */
  void loadBooks();
  
  /** @brief Disables or enables the transposition table globally. */
  void setTTDisabled(bool disable);
  
  /** @brief Queues a saved TT snapshot file to be loaded before the next search. */
  void setPendingTTLoadFile(const std::string& file) { pendingTTLoadFile_ = file; }

private:
  // --- UCI command handlers ---
  
  /** @brief Handles the 'uci' command (sends id and options). */
  void handleUci();
  
  /** @brief Handles the 'isready' command (initializes lazy resources). */
  void handleIsReady();
  
  /** @brief Handles the 'setoption' command to configure engine parameters. */
  void handleSetOption(std::istringstream &is);
  
  /** @brief Handles the 'ucinewgame' command (clears TT and search state). */
  void handleUciNewGame();
  
  /** @brief Handles the 'position' command to set up the board state. */
  void handlePosition(std::istringstream &is);
  
  /** @brief Handles the 'go' command to start a search with specified parameters. */
  void handleGo(std::istringstream &is);
  
  /** @brief Handles the 'ponderhit' command (converts a ponder search to a normal search). */
  void handlePonderHit();
  
  /** @brief Handles the 'stop' command to halt the current search immediately. */
  void handleStop();
  
  /** @brief Handles the 'quit' command to exit the engine process. */
  void handleQuit();
  
  /** @brief Handles the custom 'eval' command to print the static evaluation of the current position. */
  void handleEval();

  // --- Helper functions ---
  
  /** @brief Performs heavy lazy initialization of networks and threads. */
  void initializeEngine();
  
  /** @brief Formats an internal centipawn/mate score into a UCI-compliant string. */
  static std::string formatScore(int score);
  
  /** @brief Sends search progress information via 'info' UCI strings. */
  void sendInfo(int depth, int score, int nodes, int nps,
                const std::vector<Move> &pv, const Board &board);
                
  /** @brief Sends the 'bestmove' UCI string to conclude a search. */
  void sendBestMove(Move move, Move ponderMove = Move(Move::NO_MOVE));
  
  /** @brief Dumps a snapshot of the Transposition Table to disk (for debugging). */
  void saveTTSnapshotForMove(const std::string &moveUci);
  
  /** @brief Sanitizes a UCI string to be safe for filesystem filenames. */
  static std::string sanitizeForFilename(const std::string &value);
};