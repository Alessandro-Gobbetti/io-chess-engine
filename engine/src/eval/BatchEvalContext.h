/**
 * @file BatchEvalContext.h
 * @brief Batched evaluation context interface.
 *
 * Provides structures and abstract methods for evaluating multiple chess positions 
 * simultaneously, optimizing throughput for certain neural network architectures.
 * @ingroup engine
 */
#pragma once
// BatchEvalContext.h - Wrapper for search threads to use BatchEvaluator
// Provides the same IEvaluator interface but submits to batch queue

#include "BatchEvaluator.h"
#include "IEvaluator.h"

class BatchEvalContext : public IEvaluator {
private:
  BatchEvaluator &batchEvaluator_;

public:
  explicit BatchEvalContext(BatchEvaluator &batchEval)
      : batchEvaluator_(batchEval) {}

  // Submit to batch queue and wait for result
  float evaluate(const Board &board, int ply = 0) override {
    (void)ply;
    auto future = batchEvaluator_.submit(board);
    return future.get(); // Block until result is ready
  }

  // Submit pre-computed inputs to batch queue
  float evaluate(const ChessInput &input) override {
    auto future = batchEvaluator_.submit(input);
    return future.get();
  }
};
