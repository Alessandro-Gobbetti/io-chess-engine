/**
 * @file MCTS.cpp
 * @brief Missing description.
 * @ingroup engine
 */
#include "MCTS.h"
#include <algorithm>
#include <chrono>

MCTS::MCTS(IEvaluator &eval) : evalCtx_(eval) {}

Move MCTS::startSearch(Board &root, const SearchParams &params) {
  searching_ = true;
  stopFlag_ = false;
  nodes_ = 0;

  // Create root node
  root_ = std::make_unique<MCTSNode>();
  root_->expanded = true;

  // Expand root with all legal moves
  Movelist moves;
  MoveGen::generateLegalMoves(moves, root);

  if (moves.size() == 0) {
    searching_ = false;
    return Move(Move::NO_MOVE); // No legal moves
  }

  if (moves.size() == 1) {
    searching_ = false;
    return moves[0]; // Only one move
  }

  // Create children for all legal moves
  for (const Move &move : moves) {
    root_->children.push_back(std::make_unique<MCTSNode>(move, root_.get()));
  }

  auto startTime = std::chrono::steady_clock::now();
  int64_t timeLimit = params.movetime > 0
                          ? params.movetime
                          : (params.infinite ? 0 : 5000); // Default 5 seconds

  maxPlayouts_ = params.nodes > 0 ? params.nodes : 100000;

  // Main MCTS loop
  int playouts = 0;
  while (!stopFlag_ && playouts < maxPlayouts_) {
    // Check time
    if (timeLimit > 0) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - startTime)
                         .count();
      if (elapsed >= timeLimit)
        break;
    }

    // Copy board for this playout
    Board boardCopy = root;

    // Selection: traverse to leaf
    MCTSNode *leaf = selection(root_.get(), boardCopy);

    // Expansion: add children if not terminal
    if (!leaf->expanded) {
      expansion(leaf, boardCopy);
    }

    // Simulation: evaluate position with NN
    float value = simulation(boardCopy);

    // Backpropagation: update statistics
    backpropagation(leaf, value);

    playouts++;
    nodes_++;

    // Periodic info output
    if (infoCallback_ && (playouts % 1000 == 0)) {
      MCTSNode *bestChild = selectBestChild(root_.get());
      if (bestChild) {
        int score = static_cast<int>(bestChild->meanValue() * 100);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - startTime)
                           .count();
        int nps = elapsed > 0 ? static_cast<int>((nodes_ * 1000) / elapsed) : 0;

        std::vector<Move> pv;
        pv.push_back(bestChild->move);
        infoCallback_(1, score, nodes_, nps, pv);
      }
    }
  }

  // Select best move (most visits)
  MCTSNode *bestChild = selectBestChild(root_.get());
  Move bestMove = bestChild ? bestChild->move : moves[0];

  searching_ = false;
  return bestMove;
}

MCTSNode *MCTS::selection(MCTSNode *node, Board &board) {
  while (node->expanded && !node->children.empty()) {
    node = selectUCBChild(node);
    board.makeMove(node->move);
  }
  return node;
}

void MCTS::expansion(MCTSNode *node, Board &board) {
  Movelist moves;
  MoveGen::generateLegalMoves(moves, board);

  // Terminal node (checkmate or stalemate)
  if (moves.size() == 0) {
    node->expanded = true;
    return;
  }

  // Create children for all legal moves
  for (const Move &move : moves) {
    node->children.push_back(std::make_unique<MCTSNode>(move, node));
  }

  node->expanded = true;
}

float MCTS::simulation(Board &board) {
  // Use neural network evaluation directly (no caching)
  float eval = evalCtx_.evaluate(board);

  // Convert to value in [0, 1] range (from side to move perspective)
  // Assuming eval is roughly in [-1, 1] range (win probability style)
  float value = (eval + 1.0f) / 2.0f;
  value = std::max(0.0f, std::min(1.0f, value));

  return value;
}

void MCTS::backpropagation(MCTSNode *node, float value) {
  while (node != nullptr) {
    node->visits++;
    node->valueSum += value;

    // Flip value for opponent's perspective
    value = 1.0f - value;
    node = node->parent;
  }
}

MCTSNode *MCTS::selectBestChild(MCTSNode *node) const {
  if (node->children.empty())
    return nullptr;

  // Select child with most visits (robust choice)
  MCTSNode *best = nullptr;
  int maxVisits = -1;

  for (const auto &child : node->children) {
    if (child->visits > maxVisits) {
      maxVisits = child->visits;
      best = child.get();
    }
  }

  return best;
}

MCTSNode *MCTS::selectUCBChild(MCTSNode *node) const {
  if (node->children.empty())
    return nullptr;

  MCTSNode *best = nullptr;
  float bestUcb = -std::numeric_limits<float>::infinity();

  for (const auto &child : node->children) {
    float ucb = child->ucb(explorationConstant_);
    if (ucb > bestUcb) {
      bestUcb = ucb;
      best = child.get();
    }
  }

  return best;
}