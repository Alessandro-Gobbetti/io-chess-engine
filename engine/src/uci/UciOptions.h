#pragma once

/**
 * @file UciOptions.h
 * @brief Management of UCI engine options.
 *
 * Defines the configurable options for the engine that can be set by the GUI.
 */

#include <string>
#include "../search/ISearch.h"

class UciProtocol;

/**
 * @struct UciOptions
 * @brief Holds all configurable engine parameters.
 */
struct UciOptions {
  std::string modelPath;         ///< Path to neural network weights.
  std::string tbPath;            ///< Path to Syzygy tablebases.
  std::string bookPath;          ///< Path to primary Polyglot opening book.
  std::string bookPath2;         ///< Path to secondary Polyglot opening book.
  size_t hashSizeMB = 128;       ///< Transposition table size in MB.
  int numThreads = 1;            ///< Number of search threads (Lazy SMP).
  int evalThreads = 0;           ///< Number of eval threads (0 = Auto).
  bool useMCTS = false;          ///< True to use MCTS, false for Alpha-Beta.
  bool useSimpleEval = false;    ///< True to fallback to non-neural evaluation.
  bool useBook = true;           ///< True to use opening books.
  bool analyseMode = false;      ///< True if analyzing (disables time management).
  bool chess960 = false;         ///< True if playing Chess960 (Fischer Random).
  float aggression = 0.0f;       ///< Playstyle aggression modifier.
  
  // Lazy Eval options
  int lazyEvalMaxDepth = 6;      ///< Maximum depth to use lazy eval.
  int lazyEvalBaseMargin = 500;  ///< Static evaluation margin.
  int lazyEvalDepthMargin = 100; ///< Depth scaling margin.
  bool enableLazyEval = true;    ///< True to enable lazy evaluation.

  // Eval normalization
  int evalScaleBase = 750;       ///< Base scale for centipawn conversion.
  int evalScaleWeight = 25;      ///< Weight scalar for evaluation.
  bool enableEvalNormalization = true; ///< Normalizes scores.
  int nativeRebuildEveryNEvals = 0;    ///< Incremental MoE rebuild interval.

  SearchSharedData::SearchConfig searchConfig; ///< Search-specific heuristics configuration.
  
  // Tree Export Options
  bool exportTree = false;       ///< True to export the search tree (debug).
  int exportTreeDepth = 4;       ///< Depth limit for tree export.

  // TT snapshot/debug workflow
  bool autoSaveTTSnapshots = false;    ///< True to automatically save TT.
  std::string ttSnapshotDir = "tt_snapshots"; ///< Directory for TT snapshots.

  /**
   * @brief Prints all standard options to standard output.
   */
  void printOptions() const;

  /**
   * @brief Parses a single option and applies it. 
   * 
   * Triggers side effects via the UciProtocol pointer if needed (e.g. resizing hash).
   * 
   * @param name The name of the option.
   * @param value The value of the option.
   * @param uci Pointer to the UCI controller.
   */
  void setOption(const std::string& name, const std::string& value, UciProtocol* uci);
};
