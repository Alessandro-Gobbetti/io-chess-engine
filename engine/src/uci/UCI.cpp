#include "UCI.h"
#include "../eval/Evaluator.h"
#include "../eval/SimpleEvalContext.h"

#include "../tablebases/Tablebase.h"
#include <algorithm>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <iomanip>
#ifdef __EMSCRIPTEN__
#include <cstdio>
#include <emscripten.h>
#include <emscripten/threading.h>

// Futex-based stdin notification: instead of polling with sleep_for(5ms),
// the UCI loop waits on this flag. JS calls notify_stdin() after writing
// data to the SharedArrayBuffer, waking the futex immediately.
static int32_t g_stdinReady __attribute__((aligned(4))) = 0;

extern "C" EMSCRIPTEN_KEEPALIVE void notify_stdin() {
  __atomic_store_n(&g_stdinReady, 1, __ATOMIC_SEQ_CST);
  emscripten_futex_wake(&g_stdinReady, 1);
}
#endif

UciProtocol::UciProtocol(const std::string &modelPath, bool useSimpleEval,
                         const std::string &tbPath, const std::string &bookPath,
                         const std::string &bookPath2, int evalThreads)
    : modelPath_(modelPath), useSimpleEval_(useSimpleEval), tbPath_(tbPath),
      bookPath_(bookPath), bookPath2_(bookPath2), evalThreads_(evalThreads) {
  // Initialize TT with default size
  tt_.resize(hashSizeMB_);

  // Initialize tablebases if path provided
  if (!tbPath_.empty()) {
    Tablebase::init(tbPath_);
  }

  // Initialize opening book if path provided
  if (!bookPath_.empty()) {
    book_ = std::make_unique<PolyglotBook>(bookPath_);
    if (book_) {
      book_->load();
    }
  }

  // Initialize secondary opening book if path provided
  if (!bookPath2_.empty()) {
    book2_ = std::make_unique<PolyglotBook>(bookPath2_);
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
#ifdef __EMSCRIPTEN__
      // WASM non-blocking stdin: Module.stdin returns null when no data
      // is available, causing std::getline to fail with EOF. Clear both
      // the C++ stream and musl FILE* EOF flag, then wait for JS to
      // signal new data via notify_stdin() instead of polling.
      std::cin.clear();
      clearerr(stdin);
      // Wait on futex until JS signals data is available.
      // Timeout after 5s as a safety net (e.g. if notify missed).
      while (__atomic_load_n(&g_stdinReady, __ATOMIC_SEQ_CST) == 0) {
        emscripten_futex_wait(&g_stdinReady, 0, 5000.0);
      }
      __atomic_store_n(&g_stdinReady, 0, __ATOMIC_SEQ_CST);
      continue;
#else
      break; // Native: EOF on stdin means quit
#endif
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
          if (tt_.probe(key, 0, -32000, 32000, ttScore, ttMove, ttDepth,
                        ttBound)) {
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
        } else if (subcommand == "debug") {
          std::string mode;
          is >> mode;
          bool enable = (mode == "on");
          tt_.setDebugMode(enable);
          std::cout << "TT Debug Mode: " << (enable ? "ON" : "OFF")
                    << std::endl;
        }
      }
    }
  }
}

void UciProtocol::handleUci() {
  std::cout << "id name " << ENGINE_NAME << std::endl;
  std::cout << "id author " << ENGINE_AUTHOR << std::endl;

  // Options
  std::cout << "option name Hash type spin default 128 min 1 max 4096"
            << std::endl;
  std::cout << "option name Threads type spin default 1 min 1 max 128"
            << std::endl;
  std::cout << "option name EvalThreads type spin default 0 min 0 max 128"
            << std::endl;
  std::cout << "option name ModelPath type string default " << modelPath_
            << std::endl;
  std::cout << "option name SearchType type combo default Negamax var Negamax "
               "var MCTS"
            << std::endl;
  std::cout
      << "option name EvalType type combo default Neural var Neural var Simple"
      << std::endl;
  std::cout << "option name EvalDevice type combo default Auto var Auto var "
               "CPU var GPU"
            << std::endl;
  std::cout << "option name OwnBook type check default true" << std::endl;
  std::cout << "option name BookPath type string default " << bookPath_
            << std::endl;
  std::cout << "option name BookPath2 type string default " << bookPath2_
            << std::endl;
  std::cout << "option name SyzygyPath type string default " << tbPath_
            << std::endl;
  std::cout << "option name Aggression type spin default 0 min -100 max 100"
            << std::endl;
  std::cout << "option name EvalNormalizationBase type spin default 750 min 0 max 2000"
            << std::endl;
  std::cout << "option name EvalNormalizationWeight type spin default 25 min 1 max 200"
            << std::endl;
  std::cout << "option name EnableEvalNormalization type check default true"
            << std::endl;
  std::cout << "option name DisableTT type check default false" << std::endl;
  std::cout << "option name LazyEvalMaxDepth type spin default 6 min 0 max 10"
            << std::endl;
  std::cout
      << "option name LazyEvalBaseMargin type spin default 500 min 0 max 2000"
      << std::endl;
  std::cout
      << "option name LazyEvalDepthMargin type spin default 100 min 0 max 500"
      << std::endl;
  std::cout << "option name EnableLazyEval type check default "
            << enableLazyEval_ << std::endl;
  std::cout << "option name UCI_Chess960 type check default false"
            << std::endl;

  // Search Configuration
  std::cout << "option name RazorMarginD1 type spin default 350 min 0 max 2000"
            << std::endl;
  std::cout << "option name RazorMarginD2 type spin default 700 min 0 max 2000"
            << std::endl;
  std::cout << "option name RazorMarginD3 type spin default 900 min 0 max 2000"
            << std::endl;
  std::cout << "option name FutilityMargin type spin default 200 min 0 max 2000"
            << std::endl;
  std::cout << "option name ReverseFutilityMargin type spin default 120 min 0 "
               "max 2000"
            << std::endl;
  std::cout << "option name LMPBase type spin default 2 min 0 max 10"
            << std::endl;
  std::cout << "option name LMRThreshold type spin default 4 min 0 max 10"
            << std::endl;
  std::cout << "option name NullMoveR type spin default 3 min 0 max 10"
            << std::endl;
  std::cout << "option name DeltaMargin type spin default 200 min 0 max 2000"
            << std::endl;
  std::cout << "option name SingularMargin type spin default 50 min 0 max 2000"
            << std::endl;
  std::cout << "option name SingularMinDepth type spin default 6 min 0 max 20"
            << std::endl;
  std::cout << "option name InternalPruningMargin type spin default 150 min 0 "
               "max 2000"
            << std::endl;
  std::cout << "option name CorrWeight type spin default 4 min 0 max 25"
            << std::endl;
  std::cout << "option name EnableCorrHist type check default true"
            << std::endl;

  // Tunable Parameters
  std::cout << "option name RazorMarginD1 type spin default 350 min 0 max 2000"
            << std::endl;
  std::cout << "option name RazorMarginD2 type spin default 700 min 0 max 2000"
            << std::endl;
  std::cout << "option name RazorMarginD3 type spin default 900 min 0 max 2000"
            << std::endl;
  std::cout << "option name FutilityMargin type spin default 200 min 0 max 2000"
            << std::endl;
  std::cout << "option name ReverseFutilityMargin type spin default 120 min 0 "
               "max 2000"
            << std::endl;
  std::cout << "option name LMPBase type spin default 2 min 0 max 10"
            << std::endl;
  std::cout << "option name LMRThreshold type spin default 4 min 0 max 10"
            << std::endl;
  std::cout << "option name NullMoveR type spin default 3 min 0 max 10"
            << std::endl;
  std::cout << "option name DeltaMargin type spin default 200 min 0 max 2000"
            << std::endl;

  // Stats Export Options
  std::cout << "option name ExportTree type check default false" << std::endl;
  std::cout << "option name ExportTreeDepth type spin default 4 min 1 max 10" << std::endl;

  std::cout << "uciok" << std::endl;
}

void UciProtocol::handleIsReady() {
  // Initialize engine if not already done.
  // For SimpleEval mode, evaluators_ is always empty (no ONNX Evaluator);
  // check evalCtxs_ instead to avoid re-initializing every time.
  bool needsInit = useSimpleEval_ ? evalCtxs_.empty() : evaluators_.empty();
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
  std::cout << "info string Setting option " << optionName << std::endl;

  std::string value;
  while (is >> token) {
    value += (value.empty() ? "" : " ") + token;
  }
  std::cout << "info string New value: " << value << std::endl;

  // Process options
  if (optionName == "Hash") {
    hashSizeMB_ = std::stoul(value);
    tt_.resize(hashSizeMB_);
  } else if (optionName == "Threads") {
    std::cout << "info string Setting number of threads to " << value
              << std::endl;
    int newThreads = std::max(1, std::stoi(value));
    std::cout << "info string Actual number of threads set to " << newThreads
              << std::endl;
    if (newThreads != numThreads_) {
      numThreads_ = newThreads;
      tt_.setNumThreads(
          numThreads_); // Enable TT to skip ABDADA when single-threaded
      // Force re-initialization with new thread count
      evaluators_.clear();
      evalCtxs_.clear();
    }
  } else if (optionName == "EvalThreads") {
    int newEvalThreads = std::stoi(value);
    if (newEvalThreads != evalThreads_) {
      evalThreads_ = newEvalThreads;
      std::cout << "info string Setting evaluation threads to " << evalThreads_
                << (evalThreads_ == 0 ? " (Auto)" : "") << std::endl;
      // Reinitialize engine
      evaluators_.clear();
      evalCtxs_.clear();
    }
  } else if (optionName == "ModelPath") {
    modelPath_ = value;
    // Reinitialize engine with new model
    evaluators_.clear();
    evalCtxs_.clear();
  } else if (optionName == "SearchType") {
    useMCTS_ = (value == "MCTS");
  } else if (optionName == "EvalType") {
    useSimpleEval_ = (value == "Simple");
    // Reinitialize
    evaluators_.clear();
    evalCtxs_.clear();
  } else if (optionName == "EvalDevice") {
    if (value == "Auto" || value == "CPU" || value == "ONNX-CPU")
      useGPU_ = false;
    else if (value == "GPU" || value == "ONNX-GPU")
      useGPU_ = true;
    else
      std::cout << "info string Unknown EvalDevice: " << value << std::endl;

    // Reinitialize
    evaluators_.clear();
    evalCtxs_.clear();
  } else if (optionName == "OwnBook") {
    useBook_ = (value == "true");
  } else if (optionName == "BookPath") {
    bookPath_ = value;
    if (!bookPath_.empty()) {
      book_ = std::make_unique<PolyglotBook>(bookPath_);
      if (book_) {
        book_->load();
      }
    } else {
      book_.reset();
    }
  } else if (optionName == "BookPath2") {
    bookPath2_ = value;
    if (!bookPath2_.empty()) {
      book2_ = std::make_unique<PolyglotBook>(bookPath2_);
      if (book2_) {
        book2_->load();
      }
    } else {
      book2_.reset();
    }
  } else if (optionName == "Aggression") {
    // UCI spin value is -100 to 100, convert to -1.0 to 1.0
    aggression_ = std::clamp(std::stof(value) / 100.0f, -1.0f, 1.0f);
    // Update all existing eval contexts
    for (auto &ctx : evalCtxs_) {
      ctx->setAggression(aggression_);
    }
    std::cout << "info string Aggression set to " << aggression_ << std::endl;
  } else if (optionName == "EvalNormalizationBase") {
    evalScaleBase_ = std::stoi(value);
    for (auto &ctx : evalCtxs_) {
      ctx->setEvalScale(evalScaleBase_, evalScaleWeight_);
    }
    std::cout << "info string EvalNormalizationBase set to " << evalScaleBase_ << std::endl;
  } else if (optionName == "EvalNormalizationWeight") {
    evalScaleWeight_ = std::stoi(value);
    for (auto &ctx : evalCtxs_) {
      ctx->setEvalScale(evalScaleBase_, evalScaleWeight_);
    }
    std::cout << "info string EvalNormalizationWeight set to " << evalScaleWeight_ << std::endl;
  } else if (optionName == "EnableEvalNormalization") {
    enableEvalNormalization_ = (value == "true");
    for (auto &ctx : evalCtxs_) {
      ctx->setEvalNormalization(enableEvalNormalization_);
    }
    std::cout << "info string EnableEvalNormalization set to " << (enableEvalNormalization_ ? "true" : "false") << std::endl;
  } else if (optionName == "SyzygyPath") {
    // Set tablebase path
    tbPath_ = value;
    if (!tbPath_.empty()) {
      Tablebase::init(tbPath_);
    }
  } else if (optionName == "DisableTT") {
    bool disable = (value == "true");
    tt_.setDisabled(disable);
    tt_.setDisabled(disable);
    std::cout << "info string TT " << (disable ? "disabled" : "enabled")
              << std::endl;
  } else if (optionName == "LazyEvalMaxDepth") {
    lazyEvalMaxDepth_ = std::stoi(value);
  } else if (optionName == "LazyEvalBaseMargin") {
    lazyEvalBaseMargin_ = std::stoi(value);
  } else if (optionName == "LazyEvalDepthMargin") {
    lazyEvalDepthMargin_ = std::stoi(value);
  } else if (optionName == "EnableLazyEval") {
    enableLazyEval_ = (value == "true");
  } else if (optionName == "RazorMarginD1") {
    searchConfig_.razorMarginD1 = std::stoi(value);
  } else if (optionName == "RazorMarginD2") {
    searchConfig_.razorMarginD2 = std::stoi(value);
  } else if (optionName == "RazorMarginD3") {
    searchConfig_.razorMarginD3 = std::stoi(value);
  } else if (optionName == "FutilityMargin") {
    searchConfig_.futilityMargin = std::stoi(value);
  } else if (optionName == "ReverseFutilityMargin") {
    searchConfig_.reverseFutilityMargin = std::stoi(value);
  } else if (optionName == "LMPBase") {
    searchConfig_.lmpBase = std::stoi(value);
  } else if (optionName == "LMRThreshold") {
    searchConfig_.lmrThreshold = std::stoi(value);
  } else if (optionName == "NullMoveR") {
    searchConfig_.nullMoveR = std::stoi(value);
  } else if (optionName == "DeltaMargin") {
    searchConfig_.deltaMargin = std::stoi(value);
  } else if (optionName == "SingularMargin") {
    searchConfig_.singularMargin = std::stoi(value);
  } else if (optionName == "SingularMinDepth") {
    searchConfig_.singularMinDepth = std::stoi(value);
  } else if (optionName == "InternalPruningMargin") {
    searchConfig_.internalPruningMargin = std::stoi(value);
  } else if (optionName == "CorrWeight") {
    searchConfig_.corrWeight = std::stoi(value);
    std::cout << "info string CorrWeight set to " << searchConfig_.corrWeight << std::endl;
  } else if (optionName == "EnableCorrHist") {
    searchConfig_.enableCorrHist = (value == "true");
    std::cout << "info string CorrHist " << (searchConfig_.enableCorrHist ? "enabled" : "disabled") << std::endl;
  } else if (optionName == "UCI_Chess960") {
    chess960_ = (value == "true");
    std::cout << "info string Chess960 mode " << (chess960_ ? "enabled" : "disabled")
              << std::endl;
  } else if (optionName == "UCI_AnalyseMode") {
    analyseMode_ = (value == "true");
  } else if (optionName == "ExportTree") {
    exportTree_ = (value == "true");
  } else if (optionName == "ExportTreeDepth") {
    exportTreeDepth_ = std::stoi(value);
  } else {
    std::cout << "info string Unknown option: " << optionName << std::endl;
  }
}

void UciProtocol::handleUciNewGame() {
  tt_.clear();
  consecutiveBookMisses_ = 0;
  chess960_ = false;  // Reset to standard chess by default
}

void UciProtocol::handlePosition(std::istringstream &is) {
  std::string token;
  is >> token;

  if (token == "startpos") {
    currentBoard_ = Board(chess::constants::STARTPOS, chess960_);
    consecutiveBookMisses_ = 0; // Reset misses for each new sequence
    is >> token; // Consume "moves" if present
  } else if (token == "fen") {
    std::string fen;
    while (is >> token && token != "moves") {
      fen += (fen.empty() ? "" : " ") + token;
    }
    currentBoard_ = Board(fen, chess960_);
  }

  // Apply moves
  while (is >> token) {
    Move move = chess::uci::uciToMove(currentBoard_, token);
    if (move.move() != Move::NO_MOVE) {
      currentBoard_.makeMove(move);
      // DEBUG: Log applied move
      // std::cout << "info string DEBUG: Applied move " << token
      //           << " New FEN: " << currentBoard_.getFen() << std::endl;
    } else {
      std::cerr << "Invalid move in position command: " << token << std::endl;
      std::cout << "info string DEBUG: Failed to parse move " << token
                << " FEN: " << currentBoard_.getFen() << std::endl;
    }
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
  bool needsInit = useSimpleEval_ ? evalCtxs_.empty() : evaluators_.empty();
  if (needsInit) {
    initializeEngine();
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
          if (pMove != Move() && pMove.move() != Move::NO_MOVE) {
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
  ponderMove_ = Move(); // Clear previous ponder move
  
  // Store params for potential ponderhit use
  if (params.ponder) {
    ponderParams_ = params;
  }

  // Check for book move (skip during pondering - want full analysis from
  // opponent's perspective)
  if (useBook_ && !params.ponder) {
    // Stop searching books if we have missed too many times consecutively
    if (analyseMode_ || consecutiveBookMisses_ < MAX_BOOK_MISSES) {
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
        if (!analyseMode_) {
          // Normal play: return the book move immediately
          sendBestMove(bookMove);
          std::cout << "info string Book move found in " << bookSource << ": "
                    << chess::uci::moveToSan(currentBoard_, bookMove)
                    << std::endl;
          return;
        }
        // In analyse mode: probe() already printed candidates.
        // Fall through to the normal search for proper evaluation.
      } else {
        // Both books missed
        if (!analyseMode_) {
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
  if (useSimpleEval_) {
     hasEvaluators = !evalCtxs_.empty();
  }
  
  if (!hasEvaluators || evalCtxs_.size() < static_cast<size_t>(numThreads_)) {
    std::cerr << "Engine not ready - model not loaded or thread count mismatch"
              << std::endl;
    std::cout << "bestmove 0000" << std::endl;
    return;
  }

  // Prepare Workers
  if (useMCTS_) {
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
        ponderMove_ = Move();
      }
    });
    searcher_ = std::move(mcts);

    // Start MCTS search
    searchRunning_ = true;
    // CRITICAL FIX: Capture searchBoard to avoid race with position commands
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
      searchRunning_ = false;
    });

  } else {
    // --- Negamax (Lazy SMP) ---

    // 1. Create Shared Data
    auto searchShared = std::make_shared<SearchSharedData>();
    searchData_ = searchShared; // Store for handlePonderHit() access
    // Set tuning params
    searchShared->lazyEvalMaxDepth = lazyEvalMaxDepth_;
    searchShared->lazyEvalBaseMargin = lazyEvalBaseMargin_;
    searchShared->lazyEvalDepthMargin = lazyEvalDepthMargin_;
    searchShared->enableLazyEval = enableLazyEval_;
    searchShared->config = searchConfig_;
    
    // LAZY SMP: Store thread count for helper coordination
    searchShared->numThreads = numThreads_;

    // Tree Export Shared Config
    searchShared->exportTree = exportTree_;
    searchShared->exportTreeDepth = exportTreeDepth_;
    searchShared->vizTree.clear(); // Clear previous tree
    if (exportTree_) {
        // approximate allocation for depth 4 branching factor ~30
        searchShared->vizTree.reserve(exportTreeDepth_ * 50000); 
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
            ponderMove_ = Move();
          }
        });

    // Store Main Worker in searcher_ so handleStop() can access it
    Negamax *mainWorkerPtr = mainWorker.get();
    searcher_ = std::move(mainWorker);

    // Prepare Helpers
    std::vector<std::unique_ptr<Negamax>> helpers;

    // Create helpers for threads 1..numThreads-1
    for (int i = 1; i < numThreads_; ++i) {
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
    // CRITICAL FIX: Pass searchBoard (captured above at line 625) into the lambda.
    // Reading currentBoard_ inside the lambda is WRONG because another "position"
    // command may have modified it between handleGo() return and lambda execution.
    searchThread_ = std::thread([this, params, helpers = std::move(helpers),
                                 searchShared, mainWorkerPtr, searchBoard]() mutable {
      Board board = searchBoard;

      // DEBUG: Log the FEN being searched to help diagnose position mismatches
      if (params.ponder) {
        std::cout << "info string DEBUG ponder search FEN: " << board.getFen() << std::endl;
      }

      // CRITICAL FIX: Ensure stop flag is reset BEFORE Helpers start
      searchShared->stop = false;

      // Launch Helper Threads
      std::vector<std::thread> threadPool;
      std::cout << "info string Launching " << helpers.size()
                << " helper threads" << std::endl;
      for (size_t i = 0; i < helpers.size(); ++i) {
        auto *w = helpers[i].get();
        threadPool.emplace_back([w, board, params, i]() mutable {
          // std::cout << "info string Helper " << i << " started" << std::endl;
          w->startSearch(board, params);
        });
      }

      // Run Main Search
      Move bestMove = mainWorkerPtr->startSearch(board, params);

      // Wait for Helpers
      for (auto &t : threadPool) {
        if (t.joinable())
          t.join();
      }

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
  searchData_->startTime = std::chrono::steady_clock::now();
  
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
  pondering_ = false;
  ponderResultReady_ = false;
}

void UciProtocol::handleQuit() { handleStop(); }

void UciProtocol::initializeEngine() {
  evaluators_.clear();
  evaluators_.clear();
  evalCtxs_.clear();

  try {
    std::cout << "info string Initializing engine with " << numThreads_
              << " threads..." << std::endl;

    if (useSimpleEval_) {
      // Simple eval: one Evaluator with individual contexts
      std::cout << "info string Initializing Simple Evaluation (Hand-Crafted)..." << std::endl;
      for (int i = 0; i < numThreads_; ++i) {
        evalCtxs_.push_back(std::make_unique<SimpleEvalContext>());
      }
      // Note: evaluators_ list remains empty for SimpleEval as we don't use the ONNX Evaluator wrapper
    } else {
      // Neural eval: per-thread EvalContexts (no batching - simpler & faster)
      if (modelPath_.empty()) {
        std::cerr << "Error: Model path required for Neural evaluation"
                  << std::endl;
        return;
      }

      // === SMART EVAL THREAD ALLOCATION ===
      // With shared ONNX sessions, we no longer multiply threads.
      // The session is shared across all search threads.
      
      int evalThreadsPerInstance;
      if (evalThreads_ == 0) {
        // Auto mode: smart allocation based on search threads
        int availableCores = std::thread::hardware_concurrency();
        if (availableCores <= 0) availableCores = 4;
        
        // Reserve cores for search threads, use remainder for eval
        int remainingCores = std::max(1, availableCores - numThreads_);
        
        if (numThreads_ == 1) {
          evalThreadsPerInstance = std::min(4, remainingCores);
        } else if (numThreads_ <= 2) {
          evalThreadsPerInstance = std::min(3, remainingCores);
        } else if (numThreads_ <= 4) {
          evalThreadsPerInstance = std::min(2, remainingCores);
        } else {
          evalThreadsPerInstance = 1;
        }
        
        evalThreadsPerInstance = std::max(1, evalThreadsPerInstance);
      } else {
        evalThreadsPerInstance = evalThreads_;
      }
      
      std::string deviceStr = useGPU_ ? "GPU" : "CPU";
      std::cout << "info string Using ONNX Runtime (" << deviceStr << ") backend with SHARED session" << std::endl;
      std::cout << "info string Lazy SMP (ONNX): " << numThreads_ 
                << " search threads + " << evalThreadsPerInstance 
                << " eval threads = " << (numThreads_ + evalThreadsPerInstance)
                << " total CPU threads" << std::endl;
      
      // Create ONE shared evaluator
      auto sharedEvaluator = std::make_unique<Evaluator>(modelPath_, evalThreadsPerInstance, useGPU_);
      
      // Create lightweight contexts from shared evaluator
      for (int i = 0; i < numThreads_; ++i) {
        auto ctx = sharedEvaluator->createThreadContext();
        evalCtxs_.push_back(std::move(ctx));
      }
      
      // Keep evaluator alive (owns the shared session)
      evaluators_.push_back(std::move(sharedEvaluator));
      
      std::cout << "info string Created " << numThreads_ 
                << " eval contexts sharing single ONNX session" << std::endl;
    }

    std::cout << "info string Engine initialized with " << evalCtxs_.size()
              << " evaluation contexts" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Failed to initialize engine: " << e.what() << std::endl;
    evaluators_.clear();
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
  std::cout << "info depth " << depth << " score " << formatScore(score)
            << " nodes " << nodes << " nps " << nps;

  if (!pv.empty()) {
    std::cout << " pv";
    // Output PV in UCI notation
    for (const Move &move : pv) {
      std::cout << " " << chess::uci::moveToUci(move, chess960_);
    }
  }

  std::cout << std::endl;
}

void UciProtocol::sendBestMove(Move move, Move ponderMove) {
  std::cout << "bestmove " << chess::uci::moveToUci(move, chess960_);
  if (ponderMove != Move() && ponderMove.move() != Move::NO_MOVE) {
    std::cout << " ponder " << chess::uci::moveToUci(ponderMove, chess960_);
  }
  std::cout << std::endl;
}

void UciProtocol::handleEval() {
  // Initialize if needed
  bool needsInit = useSimpleEval_ ? evalCtxs_.empty() : evaluators_.empty();
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
            << wdl.win << " " << wdl.draw << " " << wdl.loss 
            << " mate " << wdl.mate << std::endl;
}
