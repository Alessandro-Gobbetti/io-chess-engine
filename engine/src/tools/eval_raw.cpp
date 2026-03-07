// eval_raw.cpp - Evaluate a single FEN with the raw model output
// Usage:
//   ./eval_raw --model /path/to/model.onnx "fen string"
//   ./eval_raw --simple "fen string"            # use simple evaluator (no
//   model)
//
// Prints the position evaluation, then all legal moves sorted by evaluation.

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Types.h"
#include "eval/Evaluator.h"
#include "eval/IEvaluator.h"
#include "eval/SimpleEvalContext.h"

static void print_usage(const char *prog) {
  std::cerr << "Usage: " << prog << " [--model <path> | --simple] \"<fen>\""
            << std::endl;
}

struct MoveEval {
  Move move;
  float eval;
  float win, draw, loss, mate;
};

int main(int argc, char *argv[]) {
  std::string modelPath;
  bool useSimple = false;
  bool noMoves = false;
  std::string fen;

  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      modelPath = argv[++i];
    } else if (std::strcmp(argv[i], "--simple") == 0) {
      useSimple = true;
    } else if (std::strcmp(argv[i], "--no-moves") == 0) {
      noMoves = true;
    } else {
      fen = argv[i];
    }
  }

  if (!useSimple && modelPath.empty()) {
    std::cerr << "Error: provide --model <path> or use --simple" << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  if (fen.empty()) {
    std::cerr << "Error: FEN string is required" << std::endl;
    print_usage(argv[0]);
    return 1;
  }

  try {
    std::unique_ptr<Evaluator> evaluator;
    std::unique_ptr<IEvaluator> ctx;
    std::unique_ptr<SimpleEvalContext> simpleEval;

    if (useSimple) {
        ctx = std::make_unique<SimpleEvalContext>();
    } else {
        evaluator = std::make_unique<Evaluator>(modelPath, 1);
        ctx = evaluator->createThreadContext();
    }

    Board board;
    board.setFen(fen);

    // Evaluate current position
    float rootEval = ctx->evaluate(board);

    std::cout << "=== Position Analysis ===" << std::endl;
    std::cout << "FEN: " << fen << std::endl;
    std::cout << "Side to move: "
              << (board.sideToMove() == Color::WHITE ? "White" : "Black")
              << std::endl;

    // Show raw WDL if using neural eval
    /*
    EvalContext *ec = nullptr;
    if (!useSimple) {
      ec = dynamic_cast<EvalContext *>(ctx.get());
      if (ec) {
        auto wdl = ec->evaluateWDL(board);
        std::cout << "Raw WDL+Mate: W=" << std::fixed << std::setprecision(3)
                  << wdl.win << " D=" << wdl.draw << " L=" << wdl.loss
                  << " M=" << wdl.mate << std::endl;
      }
    }
    */

    std::cout << "Position eval: " << std::fixed << std::setprecision(0)
              << rootEval << " cp" << std::endl;
    std::cout << std::endl;

    if (noMoves) {
        return 0;
    }

    // Generate all legal moves
    Movelist moves;
    MoveGen::generateLegalMoves(moves, board);

    std::cout << "=== Legal Moves (" << moves.size()
              << ") sorted by eval ===" << std::endl;

    // Evaluate each move
    std::vector<MoveEval> moveEvals;
    moveEvals.reserve(moves.size());

    for (const auto &move : moves) {
      Board child = board;
      child.makeMove(move);

      MoveEval me;
      me.move = move;

      // Get WDL from opponent's view, then flip W<->L for our view
      /*
      if (ec) {
        auto wdl = ec->evaluateWDL(child);
        // Flip: opponent's win is our loss, opponent's loss is our win
        me.win = wdl.loss;
        me.draw = wdl.draw;
        me.loss = wdl.win;
        me.mate = wdl.mate;
      } else {
      */
        me.win = me.draw = me.loss = me.mate = 0.0f;
      // }

      // Evaluate from opponent's perspective, then negate
      // std::cout << "Evaluating move " << chess::uci::moveToUci(move) << "..." << std::flush;
      me.eval = -ctx->evaluate(child);
      // std::cout << " Done: " << me.eval << std::endl;

      moveEvals.push_back(me);
    }
    std::cout << "Evaluation loop complete." << std::endl;

    // Sort by eval (best first for side to move)
    std::sort(moveEvals.begin(), moveEvals.end(),
              [](const MoveEval &a, const MoveEval &b) {
                return a.eval > b.eval; // Higher is better
              });

    // Print sorted moves with WDL
    int rank = 1;
    std::cout << std::setw(4) << "Rk" << " " << std::setw(7) << std::left
              << "Move"
              << " " << std::setw(7) << std::right << "CP"
              << " | " << std::setw(6) << "Win" << " " << std::setw(6) << "Draw"
              << " " << std::setw(6) << "Loss" << " " << std::setw(6) << "Mate"
              << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    for (const auto &me : moveEvals) {
      std::cout << std::setw(3) << rank++ << ". " << std::setw(7) << std::left
                << chess::uci::moveToUci(me.move) << " " << std::setw(7)
                << std::right << std::fixed << std::setprecision(0) << me.eval
                << " | " << std::setprecision(3) << std::setw(6) << me.win
                << " " << std::setw(6) << me.draw << " " << std::setw(6)
                << me.loss << " " << std::setw(6) << me.mate << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "Failed to evaluate: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
