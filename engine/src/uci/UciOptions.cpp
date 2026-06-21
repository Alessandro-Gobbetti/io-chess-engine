/**
 * @file UciOptions.cpp
 * @brief UCI Options configuration manager.
 *
 * Parses, validates, and stores runtime configurable options exposed by the engine 
 * over the Universal Chess Interface protocol (e.g. Hash size, Threads, Hash table usage).
 * @ingroup engine
 */
#include "UciOptions.h"
#include "UCI.h"
#include <iostream>
#include <algorithm>

void UciOptions::printOptions() const {
  std::cout << "option name Hash type spin default 128 min 1 max 4096\n";
  std::cout << "option name Threads type spin default 1 min 1 max 128\n";
  std::cout << "option name EvalThreads type spin default 0 min 0 max 128\n";
  std::cout << "option name ModelPath type string default " << modelPath << "\n";
  std::cout << "option name SearchType type combo default Negamax var Negamax var MCTS\n";
  std::cout << "option name EvalType type combo default Neural var Neural var Simple\n";
  std::cout << "option name OwnBook type check default true\n";
  std::cout << "option name BookPath type string default " << bookPath << "\n";
  std::cout << "option name BookPath2 type string default " << bookPath2 << "\n";
  std::cout << "option name SyzygyPath type string default " << tbPath << "\n";
  std::cout << "option name Aggression type spin default 0 min -100 max 100\n";
  std::cout << "option name EvalNormalizationBase type spin default 750 min 0 max 2000\n";
  std::cout << "option name EvalNormalizationWeight type spin default 25 min 1 max 200\n";
  std::cout << "option name EnableEvalNormalization type check default true\n";
  std::cout << "option name NativeRebuildEveryNEvals type spin default 0 min 0 max 1000000\n";
  std::cout << "option name DisableTT type check default false\n";
  std::cout << "option name LazyEvalMaxDepth type spin default 6 min 0 max 10\n";
  std::cout << "option name LazyEvalBaseMargin type spin default 500 min 0 max 2000\n";
  std::cout << "option name LazyEvalDepthMargin type spin default 100 min 0 max 500\n";
  std::cout << "option name EnableLazyEval type check default " << enableLazyEval << "\n";
  std::cout << "option name UCI_Chess960 type check default false\n";
  std::cout << "option name AutoSaveTTSnapshots type check default false\n";
  std::cout << "option name TTSnapshotDir type string default " << ttSnapshotDir << "\n";
  std::cout << "option name TTLoadFile type string default\n";

  // Search Configuration
  std::cout << "option name RazorMarginD1 type spin default 350 min 0 max 2000\n";
  std::cout << "option name RazorMarginD2 type spin default 700 min 0 max 2000\n";
  std::cout << "option name RazorMarginD3 type spin default 900 min 0 max 2000\n";
  std::cout << "option name FutilityMargin type spin default 200 min 0 max 2000\n";
  std::cout << "option name ReverseFutilityMargin type spin default 120 min 0 max 2000\n";
  std::cout << "option name LMPBase type spin default 2 min 0 max 10\n";
  std::cout << "option name LMRThreshold type spin default 4 min 0 max 10\n";
  std::cout << "option name NullMoveR type spin default 3 min 0 max 10\n";
  std::cout << "option name DeltaMargin type spin default 200 min 0 max 2000\n";
  std::cout << "option name SingularMargin type spin default 50 min 0 max 2000\n";
  std::cout << "option name SingularMinDepth type spin default 6 min 0 max 20\n";
  std::cout << "option name InternalPruningMargin type spin default 150 min 0 max 2000\n";
  std::cout << "option name CorrWeight type spin default 4 min 0 max 25\n";
  std::cout << "option name EnableCorrHist type check default true\n";

  // Stats Export Options
  std::cout << "option name ExportTree type check default false\n";
  std::cout << "option name ExportTreeDepth type spin default 4 min 1 max 10\n";

  std::cout << std::flush;
}

void UciOptions::setOption(const std::string& optionName, const std::string& value, UciProtocol* uci) {
  if (optionName == "Hash") {
    hashSizeMB = std::stoul(value);
    if (uci) uci->resizeHash(hashSizeMB);
  } else if (optionName == "Threads") {
    std::cout << "info string Setting number of threads to " << value << std::endl;
    int newThreads = std::max(1, std::stoi(value));
    std::cout << "info string Actual number of threads set to " << newThreads << std::endl;
    if (newThreads != numThreads) {
      numThreads = newThreads;
      if (uci) uci->updateThreads(numThreads);
    }
  } else if (optionName == "EvalThreads") {
    int newEvalThreads = std::stoi(value);
    if (newEvalThreads != evalThreads) {
      evalThreads = newEvalThreads;
      std::cout << "info string Setting evaluation threads to " << evalThreads
                << (evalThreads == 0 ? " (Auto)" : "") << std::endl;
      if (uci) uci->reinitEngine();
    }
  } else if (optionName == "ModelPath") {
    modelPath = value;
    if (uci) uci->reinitEngine();
  } else if (optionName == "SearchType") {
    useMCTS = (value == "MCTS");
  } else if (optionName == "EvalType") {
    useSimpleEval = (value == "Simple");
    if (uci) uci->reinitEngine();
  } else if (optionName == "EvalDevice") {
    std::cout << "info string EvalDevice ignored: engine runtime uses native bundle evaluation only" << std::endl;
  } else if (optionName == "OwnBook") {
    useBook = (value == "true");
  } else if (optionName == "BookPath") {
    bookPath = value;
    if (uci) uci->loadBooks();
  } else if (optionName == "BookPath2") {
    bookPath2 = value;
    if (uci) uci->loadBooks();
  } else if (optionName == "Aggression") {
    aggression = std::clamp(std::stof(value) / 100.0f, -1.0f, 1.0f);
    if (uci) uci->updateEvalContexts();
    std::cout << "info string Aggression set to " << aggression << std::endl;
  } else if (optionName == "EvalNormalizationBase") {
    evalScaleBase = std::stoi(value);
    if (uci) uci->updateEvalContexts();
    std::cout << "info string EvalNormalizationBase set to " << evalScaleBase << std::endl;
  } else if (optionName == "EvalNormalizationWeight") {
    evalScaleWeight = std::stoi(value);
    if (uci) uci->updateEvalContexts();
    std::cout << "info string EvalNormalizationWeight set to " << evalScaleWeight << std::endl;
  } else if (optionName == "EnableEvalNormalization") {
    enableEvalNormalization = (value == "true");
    if (uci) uci->updateEvalContexts();
    std::cout << "info string EnableEvalNormalization set to " << (enableEvalNormalization ? "true" : "false") << std::endl;
  } else if (optionName == "NativeRebuildEveryNEvals") {
    nativeRebuildEveryNEvals = std::max(0, std::stoi(value));
    if (uci) uci->updateEvalContexts();
    std::cout << "info string NativeRebuildEveryNEvals set to " << nativeRebuildEveryNEvals << std::endl;
  } else if (optionName == "SyzygyPath") {
    tbPath = value;
    if (uci) uci->initTablebase();
  } else if (optionName == "DisableTT") {
    bool disable = (value == "true");
    if (uci) uci->setTTDisabled(disable);
    std::cout << "info string TT " << (disable ? "disabled" : "enabled") << std::endl;
  } else if (optionName == "LazyEvalMaxDepth") {
    lazyEvalMaxDepth = std::stoi(value);
  } else if (optionName == "LazyEvalBaseMargin") {
    lazyEvalBaseMargin = std::stoi(value);
  } else if (optionName == "LazyEvalDepthMargin") {
    lazyEvalDepthMargin = std::stoi(value);
  } else if (optionName == "EnableLazyEval") {
    enableLazyEval = (value == "true");
  } else if (optionName == "RazorMarginD1") {
    searchConfig.razorMarginD1 = std::stoi(value);
  } else if (optionName == "RazorMarginD2") {
    searchConfig.razorMarginD2 = std::stoi(value);
  } else if (optionName == "RazorMarginD3") {
    searchConfig.razorMarginD3 = std::stoi(value);
  } else if (optionName == "FutilityMargin") {
    searchConfig.futilityMargin = std::stoi(value);
  } else if (optionName == "ReverseFutilityMargin") {
    searchConfig.reverseFutilityMargin = std::stoi(value);
  } else if (optionName == "LMPBase") {
    searchConfig.lmpBase = std::stoi(value);
  } else if (optionName == "LMRThreshold") {
    searchConfig.lmrThreshold = std::stoi(value);
  } else if (optionName == "NullMoveR") {
    searchConfig.nullMoveR = std::stoi(value);
  } else if (optionName == "DeltaMargin") {
    searchConfig.deltaMargin = std::stoi(value);
  } else if (optionName == "SingularMargin") {
    searchConfig.singularMargin = std::stoi(value);
  } else if (optionName == "SingularMinDepth") {
    searchConfig.singularMinDepth = std::stoi(value);
  } else if (optionName == "InternalPruningMargin") {
    searchConfig.internalPruningMargin = std::stoi(value);
  } else if (optionName == "CorrWeight") {
    searchConfig.corrWeight = std::stoi(value);
  } else if (optionName == "EnableCorrHist") {
    searchConfig.enableCorrHist = (value == "true");
  } else if (optionName == "ExportTree") {
    exportTree = (value == "true");
  } else if (optionName == "ExportTreeDepth") {
    exportTreeDepth = std::stoi(value);
  } else if (optionName == "UCI_Chess960") {
    chess960 = (value == "true");
  } else if (optionName == "AutoSaveTTSnapshots") {
    autoSaveTTSnapshots = (value == "true");
  } else if (optionName == "TTSnapshotDir") {
    ttSnapshotDir = value;
  } else if (optionName == "TTLoadFile") {
    if (uci) uci->setPendingTTLoadFile(value);
  }
}
