#pragma once

/**
 * @file MCTS.h
 * @brief Monte Carlo Tree Search implementation.
 *
 * Uses the Upper Confidence Bound for Trees (UCT) formula for selection 
 * and a neural network (or standard heuristic) for evaluation.
 */

#include "../eval/IEvaluator.h"
#include "ISearch.h"
#include <cmath>
#include <memory>
#include <vector>

/**
 * @struct MCTSNode
 * @brief Represents a single node in the Monte Carlo Search Tree.
 */
struct MCTSNode {
  Move move; ///< Move that led to this node.
  MCTSNode *parent = nullptr; ///< Pointer to the parent node.
  std::vector<std::unique_ptr<MCTSNode>> children; ///< Child nodes.

  int visits = 0;           ///< Number of times this node has been visited.
  float valueSum = 0.0f;    ///< Cumulative value from all rollouts/evaluations passing through this node.
  bool expanded = false;    ///< True if the node's children have been generated.

  MCTSNode() = default;
  MCTSNode(Move m, MCTSNode *p) : move(m), parent(p) {}

  /**
   * @brief Calculates the Upper Confidence Bound (UCB1) for this node.
   * 
   * @param explorationConstant The exploration weight (e.g., sqrt(2)).
   * @return The UCB score used for selection.
   */
  float ucb(float explorationConstant) const {
    if (visits == 0)
      return std::numeric_limits<float>::infinity();
    float avgValue = valueSum / visits;
    float exploration =
        explorationConstant * std::sqrt(std::log(parent->visits) / visits);
    return avgValue + exploration;
  }

  /**
   * @brief Calculates the mean expected value of the node.
   * 
   * @return The average value score.
   */
  float meanValue() const { return visits > 0 ? valueSum / visits : 0.0f; }
};

/**
 * @class MCTS
 * @brief Monte Carlo Tree Search engine.
 * 
 * Implements the ISearch interface. Explores the game tree asymmetrically
 * by repeatedly selecting, expanding, simulating, and backpropagating.
 */
class MCTS : public ISearch {
private:
  IEvaluator &evalCtx_; ///< The evaluator used for leaf node evaluation.

  std::unique_ptr<MCTSNode> root_; ///< Root node of the search tree.
  std::atomic<bool> stopFlag_{false}; ///< Signals the search to stop.
  std::atomic<bool> searching_{false}; ///< True while a search is active.
  std::atomic<uint64_t> nodes_{0}; ///< Total nodes visited.

  // MCTS parameters
  float explorationConstant_ = 1.41f; ///< Weight of exploration in UCB formula.
  int maxPlayouts_ = 100000;          ///< Hard limit on search iterations.

  InfoCallback infoCallback_; ///< Callback for UCI info updates.

public:
  MCTS(IEvaluator &eval);

  // ISearch interface
  Move startSearch(Board &root, const SearchParams &params) override;
  void stop() override { stopFlag_ = true; }
  bool isSearching() const override { return searching_; }
  void setInfoCallback(InfoCallback callback) override { infoCallback_ = callback; }
  uint64_t getNodes() const override { return nodes_; }

  /**
   * @brief Sets the exploration constant (Cp) used in the UCB calculation.
   */
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