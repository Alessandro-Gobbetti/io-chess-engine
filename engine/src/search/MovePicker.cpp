/**
 * @file MovePicker.cpp
 * @brief Move generation, scoring, and selection logic for search.
 *
 * Implements staged move generation (Hash move, Captures, Killers, Quiet moves)
 * and scoring heuristics like MVV-LVA and History heuristics to optimize alpha-beta pruning.
 * @ingroup engine
 */
#include "MovePicker.h"
#include "Negamax.h"
#include "SearchHeuristics.h"
#include "../tablebases/Tablebase.h"
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace SearchConstants;
using namespace MoveOrderConstants;

int MovePicker::scoreMove(const Move &move, const Board &board,
                          const Negamax *search, bool isCapture) {
  if (isCapture) {
    // Promotion bonus: prioritize promotions over regular captures
    int promoBonus = 0;
    if (move.typeOf() == Move::PROMOTION) {
      PieceType pt = move.promotionType();
      if (pt == PieceType::QUEEN) {
        promoBonus = 400;
      } else if (pt == PieceType::ROOK) {
        promoBonus = 200;
      } else if (pt == PieceType::BISHOP || pt == PieceType::KNIGHT) {
        promoBonus = 100;
      }
    }

    // MVV-LVA for captures
    Piece captured;
    if (move.typeOf() == Move::ENPASSANT) {
      captured = Piece(PieceType::PAWN, board.sideToMove() == Color::WHITE
                                            ? Color::BLACK
                                            : Color::WHITE);
    } else {
      captured = board.at(move.to());
    }
    Piece attacker = board.at(move.from());

    int toSq = move.to().index();
    int capturedType =
        (captured == Piece::NONE) ? 0 : static_cast<int>(captured.type());
    int attackerType = static_cast<int>(attacker.type());

    // Add Capture History score
    int capHistory = (*captureHist_)[attackerType][toSq];

    return static_cast<int16_t>(CAPTURE_BASE + promoBonus + capturedType * 10 +
                                (capHistory / 128) - attackerType);
  } else {
    // Quiet moves: history score + thread-based diversity
    int fromSq = move.from().index();
    int toSq = move.to().index();
    int baseScore = (*history_)[fromSq][toSq];
    int piece = static_cast<int>(board.at(move.from()));

    if (ply_ >= 1) {
      Move prev1 = search->prevMove_[ply_ - 1];
      if (prev1 != Move(0)) {
        int pPiece1 = static_cast<int>(search->prevPiece_[ply_ - 1]);
        int pTo1 = prev1.to().index();
        baseScore += search->h_->contHist_[pPiece1][pTo1][piece][toSq];
      }
    }
    if (ply_ >= 2) {
      Move prev2 = search->prevMove_[ply_ - 2];
      if (prev2 != Move(0)) {
        int pPiece2 = static_cast<int>(search->prevPiece_[ply_ - 2]);
        int pTo2 = prev2.to().index();
        baseScore += search->h_->contHist_[pPiece2][pTo2][piece][toSq];
      }
    }
    if (ply_ >= 4) {
      Move prev4 = search->prevMove_[ply_ - 4];
      if (prev4 != Move(0)) {
        int pPiece4 = static_cast<int>(search->prevPiece_[ply_ - 4]);
        int pTo4 = prev4.to().index();
        baseScore += search->h_->contHist_[pPiece4][pTo4][piece][toSq];
      }
    }

    return static_cast<int16_t>(baseScore);
  }
}

Move MovePicker::nextMove(const Board &board, const Negamax *search,
                          bool inCheck) {
  // Phase 0: Queen promotions first (highest priority)
  if (phase_ == 0) {
    for (; nextQueenPromo_ < static_cast<int>(allMoves_.size());
         ++nextQueenPromo_) {
      const auto &move = allMoves_[nextQueenPromo_];
      if (move.typeOf() == Move::PROMOTION &&
          move.promotionType() == PieceType::QUEEN) {
        nextQueenPromo_++;
        return move;
      }
    }
    phase_ = 1;
  }

  // Phase 1: Return TT move (if legal and not already returned)
  if (phase_ == 1) {
    phase_ = 2;
    if (ttMove_ != Move(0)) {
      bool alreadyReturned = (ttMove_.typeOf() == Move::PROMOTION &&
                              ttMove_.promotionType() == PieceType::QUEEN);
      if (!alreadyReturned) {
        // Validate it is actually a legal move in this position
        for (const auto &m : allMoves_) {
          if (m == ttMove_)
            return ttMove_;
        }
      }
    }
  }

  // Phase 2: Good Captures
  if (phase_ == 2) {
    if (!capturesGenerated_) {
      generateAndScoreCaptures(board, search);
      capturesGenerated_ = true;
    }

    // Select captures in order, but defer bad ones using SEE
    while (nextCapture_ < captureCount_) {
      Move move = selectBestMove(captures_, captureScores_, nextCapture_,
                                 captureCount_);

      // Skip TT move if it was already returned
      if (move == ttMove_)
        continue;

      // We will use 0 here (strict no-loss).
      if (search->see(board, move) >= 0) {
        return move;
      } else {
        // Defer this capture until the very end
        badCaptures_[badCaptureCount_++] = move;
      }
    }
    phase_ = 3;
  }

  // Phase 3: Killer moves (quiet moves that recently caused cutoffs)
  if (phase_ == 3) {
    if (!killersGenerated_) {
      generateAndScoreKillers(board, search);
      killersGenerated_ = true;
    }
    while (nextKiller_ < killerCount_) {
      Move move = killerMoves_[nextKiller_++];
      if (move != ttMove_)
        return move;
    }
    phase_ = 4;
  }

  // Phase 4: Countermove
  if (phase_ == 4 && !countermoveReturned_) {
    countermoveReturned_ = true;
    if (countermove_ != Move(0) && countermove_ != ttMove_) {
      Move k1 = (*killers_)[ply_][0];
      Move k2 = (*killers_)[ply_][1];
      if (countermove_ != k1 && countermove_ != k2) {
        for (const auto &m : allMoves_) {
          if (m == countermove_)
            return countermove_;
        }
      }
    }
  }

  // Phase 5: Quiets
  if (phase_ == 4 || phase_ == 5) {
    phase_ = 5; // Ensure we stick to phase 5 if we fall through
    if (!quietsGenerated_) {
      generateAndScoreQuiets(board, search);
      quietsGenerated_ = true;
    }
    while (nextQuiet_ < quietCount_) {
      Move move =
          selectBestMove(quiets_, quietScores_, nextQuiet_, quietCount_);

      // Skip moves already returned
      if (move == ttMove_ || move == countermove_)
        continue;
      bool isKiller = false;
      for (int i = 0; i < killerCount_; ++i) {
        if (move == killerMoves_[i]) {
          isKiller = true;
          break;
        }
      }
      if (isKiller)
        continue;

      return move;
    }
    phase_ = 6;
  }

  // Phase 6: Bad Captures (Deferred from Phase 2)
  if (phase_ == 6) {
    while (nextBadCapture_ < badCaptureCount_) {
      Move move = badCaptures_[nextBadCapture_++];
      // Skip TT move (though it should have been good anyway)
      if (move != ttMove_)
        return move;
    }
  }

  return Move(0);
}

void MovePicker::generateAndScoreCaptures(const Board &board,
                                          const Negamax *search) {
  // Single pass: classify into captures, killers, quiets to avoid multiple
  // scans
  if (capturesGenerated_ && quietsGenerated_ && killersGenerated_)
    return;

  Move k1 = (*killers_)[ply_][0];
  Move k2 = (*killers_)[ply_][1];

  for (const auto &move : allMoves_) {
    if (move == ttMove_)
      continue;

    // Queen promotions are yielded in Phase 0; skip them here to avoid
    // re-adding and re-searching the same move in later phases.
    if (move.typeOf() == Move::PROMOTION &&
        move.promotionType() == PieceType::QUEEN)
      continue;

    bool isCap = search->isCapture(board, move);
    if (isCap) {
      if (captureCount_ < MAX_MOVES) {
        captureScores_[captureCount_] = scoreMove(move, board, search, true);
        captures_[captureCount_] = move;
        captureCount_++;
      }
      continue;
    }

    bool isKiller = (move == k1 || move == k2);
    if (isKiller) {
      // Deduplicate in case k1 == k2
      bool alreadyAdded = false;
      for (int i = 0; i < killerCount_; ++i) {
        if (killerMoves_[i] == move) {
          alreadyAdded = true;
          break;
        }
      }
      if (!alreadyAdded && killerCount_ < 2) {
        killerMoves_[killerCount_++] = move;
      }
      continue;
    }

    if (quietCount_ < MAX_MOVES) {
      quietScores_[quietCount_] = scoreMove(move, board, search, false);
      quiets_[quietCount_] = move;
      quietCount_++;
    }
  }

  capturesGenerated_ = true;
  killersGenerated_ = true;
  quietsGenerated_ = true;
}

void MovePicker::generateAndScoreKillers(const Board &board,
                                         const Negamax *search) {
  if (killersGenerated_)
    return;
  // Fallback: classify everything now
  generateAndScoreCaptures(board, search);
}

void MovePicker::generateAndScoreQuiets(const Board &board,
                                        const Negamax *search) {
  // Quiets are already populated during capture generation; guard for any
  // legacy call
  if (!quietsGenerated_) {
    generateAndScoreCaptures(board, search);
  }
}

// Template for fixed-size array selection
template <size_t N>
Move MovePicker::selectBestMove(std::array<Move, N> &list,
                                std::array<int, N> &scores, int &nextIdx,
                                int count) {
  if (nextIdx >= count)
    return Move(0);

  // Find best starting from nextIdx
  int bestIdx = nextIdx;
  for (int i = nextIdx + 1; i < count; ++i) {
    if (scores[i] > scores[bestIdx]) {
      bestIdx = i;
    }
  }

  // Swap and return
  std::swap(list[nextIdx], list[bestIdx]);
  std::swap(scores[nextIdx], scores[bestIdx]);
  return list[nextIdx++];
}
