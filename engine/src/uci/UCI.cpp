/**
 * @file UCI.cpp
 * @brief Missing description.
 * @ingroup engine
 */
#include "UCI.h"
#include "../eval/Evaluator.h"
#include "../eval/SimpleEvalContext.h"

#include "../tablebases/Tablebase.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <iomanip>
#include <cctype>
#include "../platform/WasmSupport.h"

namespace {

int64_t steadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int64_t elapsedSinceNs(int64_t startNs) {
  if (startNs <= 0)
    return 0;
  const int64_t nowNs = steadyNowNs();
  if (nowNs <= startNs)
    return 0;
  return (nowNs - startNs) / 1000000;
}

} // namespace

UciProtocol::UciProtocol(const std::string &modelPath, bool useSimpleEval,
                         const std::string &tbPath, const std::string &bookPath,
                         const std::string &bookPath2, int evalThreads)
    : options_{modelPath, tbPath, bookPath, bookPath2, 128, 1, evalThreads, false, useSimpleEval} {
  // Initialize TT with default size
  tt_.resize(options_.hashSizeMB);

  // Initialize tablebases if path provided
  if (!options_.tbPath.empty()) {
    Tablebase::init(options_.tbPath);
  }

  // Initialize opening book if path provided
  if (!options_.bookPath.empty()) {
    book_ = std::make_unique<PolyglotBook>(options_.bookPath);
    if (book_) {
      book_->load();
    }
  }

  // Initialize secondary opening book if path provided
  if (!options_.bookPath2.empty()) {
    book2_ = std::make_unique<PolyglotBook>(options_.bookPath2);
    if (book2_) {
      book2_->load();
    }
  }
}

UciProtocol::~UciProtocol() {
  if (searchThread_.joinable()) {
    if (searcher_)
      searcher_->stop();
    searchThread_.join();
  }
}

void UciProtocol::loop() {
  std::string line;

  while (true) {
    if (!std::getline(std::cin, line)) {
      if (platform::waitForStdin()) {
        continue;
      }
      // Native stdin closed (common for one-shot pipes). If a bounded search
      // is running, let it finish and emit bestmove before exiting. For
      // potentially unbounded searches (go infinite / ponder), force stop.
      if (searchThread_.joinable()) {
        if (searchMayRunForever_.load(std::memory_order_relaxed)) {
          if (searcher_)
            searcher_->stop();
        }
        searchThread_.join();
      }
      break; // Native: EOF on stdin means quit
    }
    std::istringstream is(line);
    std::string command;
    is >> command;

    if (command == "uci") {
      handleUci();
    } else if (command == "isready") {
      handleIsReady();
    } else if (command == "setoption") {
      handleSetOption(is);
    } else if (command == "ucinewgame") {
      handleUciNewGame();
    } else if (command == "position") {
      handlePosition(is);
    } else if (command == "go") {
      handleGo(is);
    } else if (command == "ponderhit") {
      handlePonderHit();
    } else if (command == "stop") {
      handleStop();
    } else if (command == "eval") {
      handleEval();
    } else if (command == "ttsave") {
      std::string filename;
      is >> filename;
      if (filename.empty()) {
        std::cout << "info string usage: ttsave <path>" << std::endl;
      } else if (tt_.saveFullToFile(filename)) {
        std::cout << "info string TT saved to " << filename << std::endl;
      } else {
        std::cout << "info string failed to save TT to " << filename
                  << std::endl;
      }
    } else if (command == "ttload") {
      std::string filename;
      is >> filename;
      if (filename.empty()) {
        std::cout << "info string usage: ttload <path>" << std::endl;
      } else if (tt_.loadFullFromFile(filename)) {
        std::cout << "info string TT loaded from " << filename << std::endl;
      } else {
        std::cout << "info string failed to load TT from " << filename
                  << std::endl;
      }
    } else if (command == "quit") {
      handleQuit();
      break;
    } else if (command == "d" || command == "display") {
      std::string subcommand;
      // Try to read a subcommand
      // Note: if line ends, 'is >> subcommand' sets failbit/eofbit and string
      // empty
      if (command == "display") {
        // "display" is always board
        std::cout << currentBoard_ << std::endl;
      } else {
        // command is "d", check subcommand
        is >> subcommand;

        if (subcommand.empty()) {
          // "d" alone means display
          std::cout << currentBoard_ << std::endl;
        } else if (subcommand == "tt") {
          tt_.dumpToFile("tt_dump.bin");
          std::cout << "TT dumped to tt_dump.bin" << std::endl;
        } else if (subcommand == "fen") {
          std::cout << "FEN: " << currentBoard_.getFen() << std::endl;
        } else if (subcommand == "ttprobe") {
          uint64_t key = currentBoard_.hash();
          Move ttMove;
          int ttScore;
          int ttDepth;
          Bound ttBound;
          if (tt_.probe(key, 0, ttScore, ttMove, ttDepth, ttBound)) {
            std::cout << "TT Hit! Score: " << ttScore
                      << " Move: " << chess::uci::moveToUci(ttMove)
                      << " Depth: " << ttDepth << std::endl;
          } else {
            int16_t cachedScore;
            if (tt_.probeEval(key, cachedScore)) {
              std::cout << "TT Eval Hit! Score: " << cachedScore << std::endl;
            } else {
              std::cout << "TT Miss" << std::endl;
            }
          }
        }
      }
    }
  }
}

void UciProtocol::handleUci() {
  std::cout << "id name " << ENGINE_NAME << "\n";
  std::cout << "id author " << ENGINE_AUTHOR << "\n";
  options_.printOptions();
  std::cout << "uciok" << std::endl;
}

void UciProtocol::handleIsReady() {
  // Initialize engine if not already done.
  // For SimpleEval mode, evaluators_ is always empty;
  // check evalCtxs_ instead to avoid re-initializing every time.
  bool needsInit = options_.useSimpleEval ? evalCtxs_.empty() : evaluators_.empty();
  if (needsInit) {
    initializeEngine();
  }
  std::cout << "readyok" << std::endl;
}

void UciProtocol::handleSetOption(std::istringstream &is) {
  std::string token;
  is >> token; // "name"

  std::string optionName;
  while (is >> token && token != "value") {
    optionName += (optionName.empty() ? "" : " ") + token;
  }

  std::string value;
  while (is >> token) {
    value += (value.empty() ? "" : " ") + token;
  }

  options_.setOption(optionName, value, this);
}

void UciProtocol::handleUciNewGame() {
  tt_.clear();
  consecutiveBookMisses_ = 0;
  options_.chess960 = false;  // Reset to standard chess by default
}

void UciProtocol::handlePosition(std::istringstream &is) {
  std::string token;
  is >> token;
  std::string lastAppliedMove;

  if (token == "startpos") {
    currentBoard_ = Board(chess::constants::STARTPOS, options_.chess960);
    consecutiveBookMisses_ = 0; // Reset misses for each new sequence
    is >> token; // Consume "moves" if present
  } else if (token == "fen") {
    std::string fen;
    while (is >> token && token != "moves") {
      fen += (fen.empty() ? "" : " ") + token;
    }
    currentBoard_ = Board(fen, options_.chess960);
  }

  // Apply moves
  while (is >> token) {
    Move move = chess::uci::uciToMove(currentBoard_, token);
    if (move.move() != Move::NO_MOVE) {
      currentBoard_.makeMove(move);
      lastAppliedMove = token;
    } else {
      std::cerr << "Invalid move in position command: " << token << std::endl;
      std::cout << "info string failed to parse move " << token
                << " FEN: " << currentBoard_.getFen() << std::endl;
    }
  }

  // Save one TT snapshot per position update, keyed to the most recent move.
  // GUIs resend full move history on each position command, so saving inside the
  // loop would duplicate snapshots for old moves many times.
  if (!lastAppliedMove.empty()) {
    saveTTSnapshotForMove(lastAppliedMove);
  }
}

void UciProtocol::handleGo(std::istringstream &is) {
  // Wait for any previous search to finish
  if (searchThread_.joinable()) {
    if (searcher_)
      searcher_->stop();
    searchThread_.join();
  }

  // Initialize engine if needed (same SimpleEval-aware check as handleIsReady)
  bool needsInit = options_.useSimpleEval ? evalCtxs_.empty() : evaluators_.empty();
  if (needsInit) {
    initializeEngine();
  }

  if (!pendingTTLoadFile_.empty()) {
    const std::string fileToLoad = pendingTTLoadFile_;
    pendingTTLoadFile_.clear();
    if (tt_.loadFullFromFile(fileToLoad)) {
      std::cout << "info string TT preloaded for search from " << fileToLoad
                << std::endl;
    } else {
      std::cout << "info string TT preload failed from " << fileToLoad
                << std::endl;
    }
  }

  // Parse search parameters early (needed for ponder handling)
  SearchParams params;
  std::string token;
  while (is >> token) {
    if (token == "wtime")
      is >> params.wtime;
    else if (token == "btime")
      is >> params.btime;
    else if (token == "winc")
      is >> params.winc;
    else if (token == "binc")
      is >> params.binc;
    else if (token == "movestogo")
      is >> params.movestogo;
    else if (token == "depth")
      is >> params.depth;
    else if (token == "nodes")
      is >> params.nodes;
    else if (token == "movetime")
      is >> params.movetime;
    else if (token == "infinite")
      params.infinite = true;
    else if (token == "multipv")
      is >> params.multiPV;
    else if (token == "ponder") {
      params.ponder = true;
      // Non-standard extension: support "go ponder <move>" where GUI provides
      // the predicted move inline. If present, apply it to the board.
      // Standard UCI expects the position to already include the ponder move.
      std::string ponderMoveStr;
      if (is >> ponderMoveStr) {
        // Check if it looks like a move (4-5 chars, e.g., e2e4 or e7e8q)
        if (ponderMoveStr.length() >= 4 && ponderMoveStr.length() <= 5 &&
            ponderMoveStr[0] >= 'a' && ponderMoveStr[0] <= 'h') {
          Move pMove = chess::uci::uciToMove(currentBoard_, ponderMoveStr);
          if (pMove.move() != Move::NO_MOVE &&
              pMove.move() != Move::NULL_MOVE) {
            currentBoard_.makeMove(pMove);
          }
        } else {
          // Not a move, might be another parameter - put it back
          // Unfortunately istringstream doesn't support putback easily,
          // so we need to re-parse. For simplicity, just continue.
          // This handles cases like "go ponder wtime 60000"
          if (ponderMoveStr == "wtime")
            is >> params.wtime;
          else if (ponderMoveStr == "btime")
            is >> params.btime;
          else if (ponderMoveStr == "winc")
            is >> params.winc;
          else if (ponderMoveStr == "binc")
            is >> params.binc;
        }
      }
    }
  }

  // Reset pondering state
  pondering_ = params.ponder;
  ponderResultReady_ = false;
  ponderMove_ = Move(Move::NO_MOVE); // Clear previous ponder move
  
  // Store params for potential ponderhit use
  if (params.ponder) {
    ponderParams_ = params;
  }

  // Track whether this search might run forever without explicit stop.
  searchMayRunForever_.store(params.infinite || params.ponder,
                             std::memory_order_relaxed);

  // Check for book move (skip during pondering - want full analysis from
  // opponent's perspective)
  if (options_.useBook && !params.ponder) {
    // Stop searching books if we have missed too many times consecutively
    if (options_.analyseMode || consecutiveBookMisses_ < MAX_BOOK_MISSES) {
      chess::Move bookMove = chess::Move::NULL_MOVE;
      std::string bookSource;

      // 1. Probe Primary Book
      if (book_) {
        bookMove = book_->probe(currentBoard_, "Book 1");
        if (bookMove != chess::Move::NULL_MOVE) {
          bookSource = "Book 1";
        }
      }

      // 2. Probe Secondary Book if Primary missed
      if (bookMove == chess::Move::NULL_MOVE && book2_) {
        bookMove = book2_->probe(currentBoard_, "Book 2");
        if (bookMove != chess::Move::NULL_MOVE) {
          bookSource = "Book 2";
        }
      }

      if (bookMove != chess::Move::NULL_MOVE) {
        consecutiveBookMisses_ = 0; // Reset on hit
        if (!options_.analyseMode) {
          // Normal play: return the book move immediately
          sendBestMove(bookMove, Move(Move::NO_MOVE));
          std::cout << "info string Book move found in " << bookSource << ": "
                    << chess::uci::moveToSan(currentBoard_, bookMove)
                    << std::endl;
          return;
        }
        // In analyse mode: probe() already printed candidates.
        // Fall through to the normal search for proper evaluation.
      } else {
        // Both books missed
        if (!options_.analyseMode) {
          consecutiveBookMisses_++;
          if (consecutiveBookMisses_ >= MAX_BOOK_MISSES) {
            std::cout << "info string Book moves exhausted. Stopping book "
                         "probing for this game."
                      << std::endl;
          }
        }
      }
    }
  }

  // Check if engine is ready
  bool hasEvaluators = !evaluators_.empty();
  if (options_.useSimpleEval) {
     hasEvaluators = !evalCtxs_.empty();
  }
  
  if (!hasEvaluators || evalCtxs_.size() < static_cast<size_t>(options_.numThreads)) {
    std::cerr << "Engine not ready - model not loaded or thread count mismatch"
              << std::endl;
    std::cout << "bestmove 0000" << std::endl;
    return;
  }

  // Snapshot baselines so all reported metrics are per-search deltas.
  searchBaseFullRebuilds_ = 0;
  for (const auto &ctx : evalCtxs_) {
    if (ctx) {
      searchBaseFullRebuilds_ += ctx->getFullRebuilds();
    }
  }
  searchBaseTbHits_ = 0;
  searchBaseTtHits_ = 0;

  // Prepare Workers
  if (options_.useMCTS) {
    // MCTS doesn't use SearchSharedData counters; avoid leaking stale values
    // from previous Negamax searches.
    searchData_.reset();

    // MCTS remains single-threaded for now as its parallelization logic differs
    // Use first context
    auto mcts = std::make_unique<MCTS>(*evalCtxs_[0]);
    // Capture board BY VALUE at search start to avoid race condition
    Board searchBoard = currentBoard_;
    mcts->setInfoCallback([this, searchBoard](int d, int s, int n, int nps,
                                              const std::vector<Move> &pv) {
      sendInfo(d, s, n, nps, pv, searchBoard);
      // Store the ponder move (2nd move in PV) for later
      if (pv.size() >= 2) {
        ponderMove_ = pv[1];
      } else {
        ponderMove_ = Move(Move::NO_MOVE);
      }
    });
    searcher_ = std::move(mcts);

    // Start MCTS search
    searchRunning_ = true;
    // Capture searchBoard to avoid race condition with position commands
    searchThread_ = std::thread([this, params, searchBoard]() {
      Board board = searchBoard;
      Move bestMove = searcher_->startSearch(board, params);

      if (params.ponder) {
        std::lock_guard<std::mutex> lk(ponderMutex_);
        ponderResultMove_ = bestMove;
        ponderResultReady_ = true;
      } else {
        sendBestMove(bestMove, ponderMove_);
      }
      searchMayRunForever_.store(false, std::memory_order_relaxed);
      searchRunning_ = false;
    });

  } else {
    // --- Negamax (Lazy SMP) ---

    // 1. Create Shared Data
    auto searchShared = std::make_shared<SearchSharedData>();
    searchData_ = searchShared; // Store for handlePonderHit() access
    // Set tuning params
    searchShared->lazyEvalMaxDepth = options_.lazyEvalMaxDepth;
    searchShared->lazyEvalBaseMargin = options_.lazyEvalBaseMargin;
    searchShared->lazyEvalDepthMargin = options_.lazyEvalDepthMargin;
    searchShared->enableLazyEval = options_.enableLazyEval;
    searchShared->config = options_.searchConfig;

    searchBaseTbHits_ = searchShared->tbHits.load(std::memory_order_relaxed);
    searchBaseTtHits_ = searchShared->ttHits.load(std::memory_order_relaxed);
    
    // LAZY SMP: Store thread count for helper coordination
    searchShared->numThreads = options_.numThreads;

    // Tree Export Shared Config
    searchShared->exportTree = options_.exportTree;
    searchShared->exportTreeDepth = options_.exportTreeDepth;
    searchShared->vizTree.clear(); // Clear previous tree
    if (options_.exportTree) {
        // approximate allocation for depth 4 branching factor ~30
        searchShared->vizTree.reserve(options_.exportTreeDepth * 50000); 
    }

    // 2. Create Workers using PERSISTENT contexts

    // Main Worker (Thread 0)
    auto mainWorker = std::make_unique<Negamax>(
        *evalCtxs_[0], tt_, searchShared, true, 0); // threadId=0

    // Capture board BY VALUE at search start to avoid race condition
    Board searchBoard = currentBoard_;
    mainWorker->setInfoCallback(
        [this, searchBoard](int d, int s, int n, int nps,
                            const std::vector<Move> &pv) {
          sendInfo(d, s, n, nps, pv, searchBoard);
          // Store the ponder move (2nd move in PV) for later
          if (pv.size() >= 2) {
            ponderMove_ = pv[1];
          } else {
            ponderMove_ = Move(Move::NO_MOVE);
          }
        });

    // Store Main Worker in searcher_ so handleStop() can access it
    Negamax *mainWorkerPtr = mainWorker.get();
    searcher_ = std::move(mainWorker);

    // Prepare Helpers
    std::vector<std::unique_ptr<Negamax>> helpers;

    // Create helpers for threads 1..numThreads-1
    for (int i = 1; i < options_.numThreads; ++i) {
      if (i >= evalCtxs_.size())
        break; // Safety check

      auto helper = std::make_unique<Negamax>(*evalCtxs_[i], tt_, searchShared,
                                              false, i); // threadId=i
      helpers.push_back(std::move(helper));
    }

    // 3. Start Search Thread
    searchRunning_ = true;

    // NOTE: We do NOT move contexts. They stay in UciProtocol.
    // They are thread-safe or thread-local by design (one per thread).
    // Pass searchBoard into the lambda.
    // Reading currentBoard_ inside the lambda is WRONG because another "position"
    // command may have modified it between handleGo() return and lambda execution.
    searchThread_ = std::thread([this, params, helpers = std::move(helpers),
                                 searchShared, mainWorkerPtr, searchBoard]() mutable {
      Board board = searchBoard;

      // Launch Helper Threads
      std::vector<std::thread> threadPool;
      for (size_t i = 0; i < helpers.size(); ++i) {
        auto *w = helpers[i].get();
        threadPool.emplace_back([w, board, params, i]() mutable {
          w->startSearch(board, params);
        });
      }

      // Run Main Search
      Move bestMove = mainWorkerPtr->startSearch(board, params);

      // Ensure helpers terminate once main result is known.
      searchShared->stop.store(true, std::memory_order_relaxed);

      // Output Result
      if (params.ponder) {
        std::lock_guard<std::mutex> lk(ponderMutex_);
        ponderResultMove_ = bestMove;
        ponderResultReady_ = true;
      } else {
        // --- Export Tree Data ---
        if (searchShared->exportTree && !searchShared->vizTree.empty()) {
            // Helper to convert PruneReason to string
            auto pruneReasonToStr = [](SearchSharedData::PruneReason r) -> const char* {
              using PR = SearchSharedData::PruneReason;
              switch(r) {
                case PR::NONE: return "none";
                case PR::LMP: return "lmp";
                case PR::FUTILITY: return "futility";
                case PR::SEE: return "see";
                case PR::DRAW: return "draw";
                case PR::RAZOR: return "razor";
                case PR::REVERSE_FUTILITY: return "rfp";
                case PR::NULL_MOVE: return "null";
                case PR::INTERNAL: return "internal";
                case PR::LAZY_EVAL: return "lazy";
                case PR::TB_CUTOFF: return "tb";
                case PR::TT_CUTOFF: return "tt";
                case PR::BETA_CUTOFF: return "beta";
                case PR::DEPTH_LIMIT: return "depth";
                case PR::SINGULAR_SKIP: return "singular";
                default: return "unknown";
              }
            };

            std::stringstream ss;
            ss << "info string json_tree [";
            bool first = true;
            for (const auto& node : searchShared->vizTree) {
                if (!first) ss << ",";
                first = false;
                ss << "{\"p\":\"" << node.parentHash 
                   << "\",\"c\":\"" << node.childHash
                   << "\",\"m\":\"" << node.move << "\""
                   << ",\"se\":" << node.staticEval
                   << ",\"ss\":" << node.searchScore
                   << ",\"d\":" << node.depth
                   << ",\"ply\":" << node.ply
                   << ",\"ord\":" << node.moveOrder
                   << ",\"n\":" << node.subtreeNodes
                   << ",\"pv\":" << (node.isPV ? "true" : "false")
                   << ",\"pr\":\"" << pruneReasonToStr(node.pruneReason) << "\""
                   << "}";
            }
            ss << "]";
            std::cout << ss.str() << std::endl;
        }
        sendBestMove(bestMove, ponderMove_);
      }

      // Join helpers after emitting bestmove so UCI stop responses are immediate.
      for (auto &t : threadPool) {
        if (t.joinable())
          t.join();
      }

      searchMayRunForever_.store(false, std::memory_order_relaxed);
      searchRunning_ = false;
    });
  }
}

void UciProtocol::handlePonderHit() {
  // Ponderhit means opponent played the predicted move.
  // Instead of stopping, we switch from infinite ponder mode to time-managed mode.
  // The search continues with proper time allocation.
  //
  // IMPORTANT: We use REDUCED time on ponderhit because:
  // 1. We've already been searching this position during ponder
  // 2. TT is already filled with valuable entries
  // 3. We likely have a good move from the ponder search
  // This allows us to save time for later moves.
  
  pondering_ = false;
  
  // Check if we have time params from the original go ponder command
  bool hasTimeParams = (ponderParams_.wtime > 0 || ponderParams_.btime > 0 || 
                        ponderParams_.movetime > 0);
  
  if (!hasTimeParams || !searchData_) {
    // No time info available - fall back to immediate stop behavior
    if (searcher_) {
      searcher_->stop();
    }
    if (searchThread_.joinable()) {
      searchThread_.join();
    }
    if (ponderResultReady_) {
      sendBestMove(ponderResultMove_, ponderMove_);
      ponderResultReady_ = false;
    }
    return;
  }
  
  // Calculate time allocation for the position
  TimeManager timeManager;
  float phase = 0.5f;  // Reasonable middle-game estimate
  float evalCp = 0.0f; // Neutral estimate
  int moveNumber = currentBoard_.fullMoveNumber();
  Color side = currentBoard_.sideToMove();
  
  // Remove ponder flag so TimeManager calculates real time limits
  SearchParams timedParams = ponderParams_;
  timedParams.ponder = false;
  timedParams.infinite = false;
  
  auto allocation = timeManager.calculate(timedParams, side, phase, evalCp, moveNumber);
  
  // PONDERHIT TIME REDUCTION:
  // Since we've already been pondering, use only 40% of normal time allocation.
  // This gives us a "bonus" from correct prediction while still allowing
  // some additional search to verify/improve the move.
  // 40% is aggressive but justified because:
  // - TT is warm with this exact position
  // - We've likely reached good depth already
  // - Saving time compounds over many moves
  constexpr float PONDERHIT_TIME_FACTOR = 0.40f;
  allocation.softLimit = static_cast<int64_t>(allocation.softLimit * PONDERHIT_TIME_FACTOR);
  allocation.hardLimit = static_cast<int64_t>(allocation.hardLimit * PONDERHIT_TIME_FACTOR);
  
  // Ensure minimum search time (at least 50ms to verify move)
  allocation.softLimit = std::max(allocation.softLimit, static_cast<int64_t>(50));
  allocation.hardLimit = std::max(allocation.hardLimit, static_cast<int64_t>(100));
  
  // Update the shared search data with new time limits
  // The search will pick these up on its next time check (every 128 nodes)
  searchData_->softLimit = allocation.softLimit;
  searchData_->hardLimit = allocation.hardLimit;
  // Reset start time to now since we're starting the "real" search
  searchData_->startTimeNs.store(steadyNowNs(), std::memory_order_relaxed);
  
  std::cout << "info string ponderhit: reduced time (soft="
            << allocation.softLimit << "ms hard=" << allocation.hardLimit 
            << "ms, " << static_cast<int>(PONDERHIT_TIME_FACTOR * 100) << "% of normal)"
            << std::endl;
  
  // Now wait for the search to complete naturally (it will stop when time runs out)
  if (searchThread_.joinable()) {
    searchThread_.join();
  }
  
  // The search thread already sent bestmove, so we're done
  // (unless ponderResultReady_ is still set from an earlier ponder result)
  if (ponderResultReady_) {
    sendBestMove(ponderResultMove_, ponderMove_);
    ponderResultReady_ = false;
  }
}

void UciProtocol::handleStop() {
  if (searcher_) {
    searcher_->stop();
  }
  if (searchThread_.joinable()) {
    searchThread_.join();
  }

  searchMayRunForever_.store(false, std::memory_order_relaxed);
  pondering_ = false;
  ponderResultReady_ = false;
}

void UciProtocol::handleQuit() { handleStop(); }

void UciProtocol::initializeEngine() {
  evaluators_.clear();
  evalCtxs_.clear();

  try {
    std::cout << "info string Initializing engine with " << options_.numThreads
              << " threads..." << std::endl;

    if (options_.useSimpleEval) {
      // Simple eval: one Evaluator with individual contexts
      std::cout << "info string Initializing Simple Evaluation (Hand-Crafted)..." << std::endl;
      for (int i = 0; i < options_.numThreads; ++i) {
        evalCtxs_.push_back(std::make_unique<SimpleEvalContext>());
      }
      // Note: evaluators_ list remains empty for SimpleEval.
    } else {
      // Neural eval: per-thread EvalContexts (no batching - simpler & faster)
      if (options_.modelPath.empty()) {
        std::cerr << "Error: Model path required for Neural evaluation"
                  << std::endl;
        return;
      }

      // Keep EvalThreads behavior for compatibility with existing UCI configs.
      // Native cache backend does not spawn separate evaluator worker pools.
      
      int evalThreadsPerInstance;
      if (options_.evalThreads == 0) {
        // Auto mode: smart allocation based on search threads
        int availableCores = std::thread::hardware_concurrency();
        if (availableCores <= 0) availableCores = 4;
        
        // Reserve cores for search threads, use remainder for eval
        int remainingCores = std::max(1, availableCores - options_.numThreads);
        
        if (options_.numThreads == 1) {
          evalThreadsPerInstance = std::min(4, remainingCores);
        } else if (options_.numThreads <= 2) {
          evalThreadsPerInstance = std::min(3, remainingCores);
        } else if (options_.numThreads <= 4) {
          evalThreadsPerInstance = std::min(2, remainingCores);
        } else {
          evalThreadsPerInstance = 1;
        }
        
        evalThreadsPerInstance = std::max(1, evalThreadsPerInstance);
      } else {
        evalThreadsPerInstance = options_.evalThreads;
      }
      
      auto sharedEvaluator = std::make_unique<Evaluator>(options_.modelPath, evalThreadsPerInstance);

      std::cout << "info string Using native incremental MoE cache backend" << std::endl;
      std::cout << "info string Lazy SMP (native cache): " << options_.numThreads
                << " search threads total" << std::endl;
      
      // Create lightweight contexts from shared evaluator
      for (int i = 0; i < options_.numThreads; ++i) {
        auto ctx = sharedEvaluator->createThreadContext();
        evalCtxs_.push_back(std::move(ctx));
      }
      
      // Keep evaluator alive (owns shared model weights)
      evaluators_.push_back(std::move(sharedEvaluator));
      
      std::cout << "info string Created " << options_.numThreads
                << " native incremental eval contexts" << std::endl;
    }

    std::cout << "info string Engine initialized with " << evalCtxs_.size()
              << " evaluation contexts" << std::endl;

    for (auto &ctx : evalCtxs_) {
      ctx->setAggression(options_.aggression);
      ctx->setEvalScale(options_.evalScaleBase, options_.evalScaleWeight);
      ctx->setEvalNormalization(options_.enableEvalNormalization);
      ctx->setIncrementalRebuildInterval(options_.nativeRebuildEveryNEvals);
    }

  } catch (const std::exception &e) {
    std::cerr << "Failed to initialize engine: " << e.what() << std::endl;
    evaluators_.clear();
    evalCtxs_.clear();
  }
}

// Helper to format score (cp or mate)
std::string UciProtocol::formatScore(int score) {
  using namespace SearchConstants;

  // Real checkmate found by search
  if (score > MATE_IN_MAX) {
    int movesToMate = (MATE_SCORE - score + 1) / 2;
    return "mate " + std::to_string(movesToMate);
  } else if (score < -MATE_IN_MAX) {
    int movesToMate = (MATE_SCORE + score + 1) / 2;
    return "mate -" + std::to_string(movesToMate);
  }
  // Tablebase win (score around TB_WIN_SCORE = 31000)
  else if (score >= TB_WIN_IN_MAX) {
    // Score = TB_WIN_SCORE - (dtz + ply)
    int dtz = TB_WIN_SCORE - score;
    return "mate " + std::to_string((dtz + 1) / 2);
  } else if (score <= -TB_WIN_IN_MAX) {
    int dtz = TB_WIN_SCORE + score;
    return "mate -" + std::to_string((dtz + 1) / 2);
  } else {
    // Regular centipawn score
    return "cp " + std::to_string(score);
  }
}

void UciProtocol::sendInfo(int depth, int score, int nodes, int nps,
                           const std::vector<Move> &pv, const Board &board) {
  (void)board;

  int seldepth = 0;
  uint64_t tbHits = 0;
  uint64_t ttHits = 0;
  int64_t elapsedMs = 0;

  if (searchData_) {
    seldepth = searchData_->selDepth.load(std::memory_order_relaxed);
    const uint64_t tbHitsTotal =
      searchData_->tbHits.load(std::memory_order_relaxed);
    const uint64_t ttHitsTotal =
      searchData_->ttHits.load(std::memory_order_relaxed);
    tbHits = tbHitsTotal >= searchBaseTbHits_ ? tbHitsTotal - searchBaseTbHits_
                          : 0;
    ttHits = ttHitsTotal >= searchBaseTtHits_ ? ttHitsTotal - searchBaseTtHits_
                          : 0;
    const int64_t startNs =
      searchData_->startTimeNs.load(std::memory_order_relaxed);
    elapsedMs = elapsedSinceNs(startNs);
  }

  uint64_t fullRebuildsTotal = 0;
  for (const auto &ctx : evalCtxs_) {
    if (ctx) {
      fullRebuildsTotal += ctx->getFullRebuilds();
    }
  }
  const uint64_t fullRebuilds =
      fullRebuildsTotal >= searchBaseFullRebuilds_
          ? fullRebuildsTotal - searchBaseFullRebuilds_
          : 0;

  std::cout << "info depth " << depth << " seldepth " << seldepth
            << " score " << formatScore(score) << " nodes " << nodes
            << " nps " << nps << " tbhits " << tbHits << " time "
            << elapsedMs;

  if (!pv.empty()) {
    std::cout << " pv";
    // Output PV in UCI notation
    for (const Move &move : pv) {
      std::cout << " " << chess::uci::moveToUci(move, options_.chess960);
    }
  }

  std::cout << " string tt_hits " << ttHits << " full_rebuilds "
            << fullRebuilds;

  std::cout << std::endl;
}

void UciProtocol::sendBestMove(Move move, Move ponderMove) {
  std::cout << "bestmove " << chess::uci::moveToUci(move, options_.chess960);

  if (ponderMove.move() != Move::NO_MOVE &&
      ponderMove.move() != Move::NULL_MOVE) {
    try {
      std::cout << " ponder " << chess::uci::moveToUci(ponderMove, options_.chess960);
    } catch (const std::exception &) {
      // Ignore malformed ponder move; bestmove itself is authoritative.
    }
  }
  std::cout << std::endl;
}

void UciProtocol::saveTTSnapshotForMove(const std::string &moveUci) {
  if (!options_.autoSaveTTSnapshots)
    return;

  std::error_code ec;
  std::filesystem::create_directories(options_.ttSnapshotDir, ec);
  if (ec) {
    std::cout << "info string TT snapshot mkdir failed for " << options_.ttSnapshotDir
              << ": " << ec.message() << std::endl;
    return;
  }

  const uint64_t snapshotId = ++ttSnapshotCounter_;
  const uint64_t hash = currentBoard_.hash();
  std::ostringstream hexHash;
  hexHash << std::hex << std::setw(16) << std::setfill('0') << hash;

  const std::string side = currentBoard_.sideToMove() == Color::WHITE ? "w" : "b";
  std::ostringstream filename;
  filename << "tt_" << std::setw(6) << std::setfill('0') << snapshotId
           << "_fm" << currentBoard_.fullMoveNumber() << "_" << side
           << "_" << sanitizeForFilename(moveUci) << "_" << hexHash.str()
           << ".bin";

  const std::filesystem::path outPath =
      std::filesystem::path(options_.ttSnapshotDir) / filename.str();
  if (tt_.saveFullToFile(outPath.string())) {
    std::cout << "info string TT snapshot saved: " << outPath.string()
              << std::endl;
  } else {
    std::cout << "info string TT snapshot save failed: " << outPath.string()
              << std::endl;
  }
}

std::string UciProtocol::sanitizeForFilename(const std::string &value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::isalnum(uc) || c == '_' || c == '-') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  return out.empty() ? "move" : out;
}

void UciProtocol::handleEval() {
  // Initialize if needed
  bool needsInit = options_.useSimpleEval ? evalCtxs_.empty() : evaluators_.empty();
  if (needsInit) {
    initializeEngine();
  }

  if (evalCtxs_.empty()) {
    std::cout << "info string Engine not ready" << std::endl;
    return;
  }
  
  // Use first context (main thread)
  auto wdl = evalCtxs_[0]->evaluateWDL(currentBoard_);
  
  std::cout << "wdl " << std::fixed << std::setprecision(4) 
            << wdl.win << " " << wdl.draw << " " << wdl.loss << std::endl;
}

void UciProtocol::resizeHash(size_t mb) {
  tt_.resize(mb);
}

void UciProtocol::updateThreads(int numThreads) {
  tt_.setNumThreads(numThreads);
  evaluators_.clear();
  evalCtxs_.clear();
}

void UciProtocol::reinitEngine() {
  evaluators_.clear();
  evalCtxs_.clear();
  initializeEngine();
}

void UciProtocol::updateEvalContexts() {
  for (auto &ctx : evalCtxs_) {
    ctx->setAggression(options_.aggression);
    ctx->setEvalScale(options_.evalScaleBase, options_.evalScaleWeight);
    ctx->setEvalNormalization(options_.enableEvalNormalization);
    ctx->setIncrementalRebuildInterval(options_.nativeRebuildEveryNEvals);
  }
}

void UciProtocol::initTablebase() {
  if (!options_.tbPath.empty()) {
    Tablebase::init(options_.tbPath);
  }
}

void UciProtocol::loadBooks() {
  if (!options_.bookPath.empty()) {
    book_ = std::make_unique<PolyglotBook>(options_.bookPath);
    if (book_) book_->load();
  } else {
    book_.reset();
  }
  
  if (!options_.bookPath2.empty()) {
    book2_ = std::make_unique<PolyglotBook>(options_.bookPath2);
    if (book2_) book2_->load();
  } else {
    book2_.reset();
  }
}

void UciProtocol::setTTDisabled(bool disable) {
  tt_.setDisabled(disable);
}
