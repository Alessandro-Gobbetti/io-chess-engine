#pragma once

/**
 * @file TablebaseFallback.h
 * @brief Integration of Syzygy tablebases at the root search node.
 */

#include "ISearch.h"
#include "../eval/IEvaluator.h"
#include <optional>
#include <memory>

namespace TablebaseFallback {

/**
 * @brief Probes the tablebase at the root node.
 * 
 * If a tablebase move is found and verified (including falling back to WDL 
 * if DTZ is missing), it prints info strings, optionally calls the infoCallback,
 * sets shared->stop to true, and returns the best move.
 * 
 * @param root The root board position.
 * @param evalCtx The active evaluator context.
 * @param shared Shared data across search threads (for stop flags).
 * @param infoCallback Callback to emit UCI info strings.
 * @return std::optional<Move> The best tablebase move, or std::nullopt if normal search should proceed.
 */
std::optional<Move> probeRoot(Board& root, IEvaluator& evalCtx,
                              std::shared_ptr<SearchSharedData>& shared,
                              const InfoCallback& infoCallback);

} // namespace TablebaseFallback
