#pragma once
// MCTS.h - Monte Carlo Tree Search implementation
// Uses UCT for selection and neural network for evaluation

#include "../eval/IEvaluator.h"
#include "ISearch.h"
#include <cmath>
#include <memory>
#include <vector>

struct MCTSNode {
  Move move; // Move that led to this node
  MCTSNode *parent = nullptr;
  std::vector<std::unique_ptr<MCTSNode>> children;

  int visits = 0;
  float valueSum = 0.0f;
  bool expanded = false;

  MCTSNode() = default;
  MCTSNode(Move m, MCTSNode *p) : move(m), parent(p) {}

  // UCB1 formula for selection
  float ucb(float explorationConstant) const {
    if (visits == 0)
      return std::numeric_limits<float>::infinity();
    float avgValue = valueSum / visits;
    float exploration =
        explorationConstant * std::sqrt(std::log(parent->visits) / visits);
    return avgValue + exploration;
  }

  float meanValue() const { return visits > 0 ? valueSum / visits : 0.0f; }
};

class MCTS : public ISearch {
private:
  IEvaluator &evalCtx_;

  std::unique_ptr<MCTSNode> root_;
  std::atomic<bool> stopFlag_{false};
  std::atomic<bool> searching_{false};
  std::atomic<uint64_t> nodes_{0};

  // MCTS parameters
  float explorationConstant_ = 1.41f; // sqrt(2) is common
  int maxPlayouts_ = 100000;

  InfoCallback infoCallback_;

public:
  MCTS(IEvaluator &eval);

  // ISearch interface
  Move startSearch(Board &root, const SearchParams &params) override;
  void stop() override { stopFlag_ = true; }
  bool isSearching() const override { return searching_; }
  void setInfoCallback(InfoCallback callback) override {
    infoCallback_ = callback;
  }
  uint64_t getNodes() const override { return nodes_; }

  // MCTS-specific settings
  void setExplorationConstant(float c) { explorationConstant_ = c; }

private:
  // Core MCTS phases
  MCTSNode *selection(MCTSNode *node, Board &board);
  void expansion(MCTSNode *node, Board &board);
  float simulation(Board &board);
  void backpropagation(MCTSNode *node, float value);

  // Utilities
  MCTSNode *selectBestChild(MCTSNode *node) const;
  MCTSNode *selectUCBChild(MCTSNode *node) const;
};