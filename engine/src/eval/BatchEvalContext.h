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
  float evaluate(const Board &board) override {
    auto future = batchEvaluator_.submit(board);
    return future.get(); // Block until result is ready
  }

  // Submit pre-computed inputs to batch queue
  float evaluate(const ChessInput &input) override {
    auto future = batchEvaluator_.submit(input);
    return future.get();
  }
};
