/**
 * @file TablebaseFallback.cpp
 * @brief Missing description.
 * @ingroup engine
 */
#include "TablebaseFallback.h"
#include "../tablebases/Tablebase.h"
#include "../uci/UCI.h"
#include <iostream>
#include <vector>
#include <algorithm>

namespace TablebaseFallback {

std::optional<Move> probeRoot(Board& root, IEvaluator& evalCtx,
                              std::shared_ptr<SearchSharedData>& shared,
                              const InfoCallback& infoCallback) {
  if (!Tablebase::available(root)) {
    return std::nullopt;
  }

  auto tbResult = Tablebase::probeRoot(root);
  if (tbResult) {
    Move tbMove = std::get<0>(*tbResult);
    Tablebase::WDL wdl = std::get<1>(*tbResult);
    int dtz = std::get<2>(*tbResult);

    // Note: DTZ does not distinguish mate speed.
    // For promotion moves, check all 4 promotion types, keep winning ones,
    // then use NN eval to pick the best (fastest mate).
    if (wdl == Tablebase::WDL::WIN && tbMove.typeOf() == Move::PROMOTION) {
      std::vector<std::pair<Move, int>> winningPromos; // (move, eval)

      const PieceType promoTypes[] = {PieceType::QUEEN, PieceType::ROOK,
                                      PieceType::BISHOP, PieceType::KNIGHT};

      for (PieceType pt : promoTypes) {
        Move promo =
            Move::make<Move::PROMOTION>(tbMove.from(), tbMove.to(), pt);

        // Check if this promotion is winning in TB
        Board testBoard = root;
        testBoard.makeMove(promo);
        auto promoWdl = Tablebase::probeWDL(testBoard);

        // If winning (opponent's view = LOSS), add to candidates
        if (promoWdl && *promoWdl == Tablebase::WDL::LOSS) {
          // Evaluate with NN to find best
          int eval = evalCtx.evaluate(testBoard);
          // Negate since it's opponent's turn after our move
          winningPromos.push_back({promo, -eval});
        }
      }

      // Pick promotion with best eval (highest = fastest mate typically)
      if (!winningPromos.empty()) {
        auto best = std::max_element(
            winningPromos.begin(), winningPromos.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });

        if (best->first != tbMove) {
          std::cout << "info string TB: eval picked "
                    << chess::uci::moveToUci(best->first) << " (eval "
                    << best->second << ") over TB suggestion" << std::endl;
        }
        tbMove = best->first;
      }
    }

    // Found TB move - check for repetition
    bool cxRepetition = false;
    {
      Board testBoard = root;
      testBoard.makeMove(tbMove);
      // Strict 2-fold repetition check for safety in winning lines
      if (testBoard.isRepetition(1)) {
        cxRepetition = true;
      }
    }

    bool acceptTbMove = true;
    if (cxRepetition) {
      if (wdl == Tablebase::WDL::WIN || wdl == Tablebase::WDL::CURSED_WIN) {
        std::cout << "info string TB: Root move "
                  << chess::uci::moveToUci(tbMove)
                  << " leads to repetition. Rejecting to preserve WIN."
                  << std::endl;
        acceptTbMove = false;
      }
    }

    if (acceptTbMove) {
      std::cout << "info string TB: hit at root, WDL="
                << static_cast<int>(wdl) << " DTZ=" << dtz << std::endl;

      if (infoCallback) {
        std::vector<Move> pv = {tbMove};
        // Use WDL + DTZ for score
        int tbScore;
        if (wdl == Tablebase::WDL::WIN) {
          tbScore = SearchConstants::TB_WIN_SCORE - (dtz * 2);
        } else if (wdl == Tablebase::WDL::LOSS) {
          tbScore = -SearchConstants::TB_WIN_SCORE + (dtz * 2);
        } else if (wdl == Tablebase::WDL::CURSED_WIN) {
          tbScore = 100;
        } else if (wdl == Tablebase::WDL::BLESSED_LOSS) {
          tbScore = -100;
        } else {
          tbScore = SearchConstants::DRAW_SCORE;
        }
        infoCallback(0, tbScore, 0, 0, pv);
      }

      // Stop other threads immediately
      if (shared) shared->stop = true;
      return tbMove;
    }
  }

  // Fallback: If probeRoot failed OR we rejected the move due to repetition
  {
    // probeRoot failed - try WDL-only fallback
    // This happens when DTZ files are missing/corrupt but WDL files work, or
    // move rejected.
    auto wdl = Tablebase::probeWDL(root);
    if (wdl &&
        (*wdl == Tablebase::WDL::WIN || *wdl == Tablebase::WDL::LOSS)) {

      // Test each legal move and find one that maintains winning/losing
      // status
      Movelist moves;
      MoveGen::generateLegalMoves(moves, root);

      Move bestTbMove = Move(0);
      int bestEval = -SearchConstants::INFINITE;

      // Prefer "zeroing" moves (pawn moves / captures) when we don't have
      // DTZ guidance. This prevents aimless king shuffles that can drift into
      // 50-move draws even in theoretically winning TB positions.
      Move bestZeroingMove = Move(0);
      int bestZeroingEval = -SearchConstants::INFINITE;

      // If any zeroing move leads to a position we can't probe with WDL
      // (common with incomplete Syzygy sets and promotions), the WDL-only
      // fallback becomes unreliable. In that case, fall back to normal search
      // rather than returning a potentially drawish king shuffle.
      bool missingZeroingChildProbe = false;

      for (const Move &move : moves) {
        Board testBoard = root;
        testBoard.makeMove(move);

        auto childWdl = Tablebase::probeWDL(testBoard);

        // Identify moves that reset the 50-move clock.
        // - Any pawn move (including promotions)
        // - Any capture (including en passant)
        const Piece movingPiece = root.at(move.from());
        const bool isPawnMove = (movingPiece.type() == PieceType::PAWN) ||
                                (move.typeOf() == Move::PROMOTION);
        const bool isCapture = (root.at(move.to()) != Piece::NONE) ||
                               (move.typeOf() == Move::ENPASSANT);
        const bool isZeroingMove = isPawnMove || isCapture;

        if (!childWdl) {
          if (isZeroingMove) {
            missingZeroingChildProbe = true;
          }
          continue;
        }

        // If we're winning, pick move where opponent is losing
        // If we're losing, this fallback won't help much but pick least bad
        bool isGoodMove = false;
        if (*wdl == Tablebase::WDL::WIN &&
            *childWdl == Tablebase::WDL::LOSS) {
          // STRICT SAFETY: Check repetition for candidate moves too!
          if (testBoard.isRepetition(1)) {
            continue; // Skip this move, it repeats!
          }
          isGoodMove = true;
        } else if (*wdl == Tablebase::WDL::LOSS &&
                   *childWdl == Tablebase::WDL::WIN) {
          isGoodMove = true; // We're losing, opponent winning - expected
        }

        if (isGoodMove) {
          // Use NN eval to break ties (prefer faster wins)
          const int eval = -evalCtx.evaluate(testBoard);

          if (isZeroingMove) {
            if (eval > bestZeroingEval) {
              bestZeroingEval = eval;
              bestZeroingMove = move;
            }
          }

          if (eval > bestEval) {
            bestEval = eval;
            bestTbMove = move;
          }
        }
      }

      // If any zeroing move couldn't be probed, don't trust the WDL-only
      // fallback (it may exclude critical promotions/captures). Let normal
      // search decide instead.
      if (missingZeroingChildProbe) {
        if (shared) {
          shared->disableTablebase.store(true, std::memory_order_relaxed);
        }
        std::cout
            << "info string TB: WDL fallback incomplete (missing child probe for zeroing move); disabling TB for this search"
            << std::endl;
      } else {
        // If we found any winning/losing zeroing move, take it.
        // Otherwise fall back to the best WDL-preserving move.
        const Move chosenTbMove = (bestZeroingMove != Move(0))
                                      ? bestZeroingMove
                                      : bestTbMove;

        if (chosenTbMove != Move(0)) {
          // SAFETY CHECK: Verify move is legal before returning
          Piece piece = root.at(chosenTbMove.from());
          if (piece == Piece::NONE || piece.color() != root.sideToMove()) {
            std::cout
                << "info string CRITICAL: TB Fallback selected illegal move "
                << chess::uci::moveToUci(chosenTbMove)
                << " (Piece: " << (int)piece.type()
                << ", Color: " << (int)piece.color() << ")"
                << ". Falling back to search." << std::endl;
            // Do not return; fall through to normal search
          } else {
            std::cout << "info string TB: WDL fallback found "
                      << chess::uci::moveToUci(chosenTbMove)
                      << ((bestZeroingMove != Move(0)) ? " (zeroing)" : "")
                      << std::endl;

            int tbScore = Tablebase::wdlToScore(*wdl, 0);
            if (infoCallback) {
              std::vector<Move> pv = {chosenTbMove};
              infoCallback(0, tbScore, 0, 0, pv);
            }

            if (shared) shared->stop = true;
            return chosenTbMove;
          }
        }
      }
    }
  }

  return std::nullopt;
}

} // namespace TablebaseFallback
