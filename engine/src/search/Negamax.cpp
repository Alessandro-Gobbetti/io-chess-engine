#include "Negamax.h"
#include "../tablebases/Tablebase.h"
#include "FeatureExtractor.hpp" // For game phase extraction
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace SearchConstants;
using namespace MoveOrderConstants;

// ============================================================================
//   MovePicker Implementation
// ============================================================================

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
                                (capHistory / 128) -
                                attackerType);
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
    for (; nextQueenPromo_ < static_cast<int>(allMoves_.size()); ++nextQueenPromo_) {
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
          if (m == ttMove_) return ttMove_;
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
      Move move = selectBestMove(captures_, captureScores_, nextCapture_, captureCount_);
      
      // Skip TT move if it was already returned
      if (move == ttMove_) continue;

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
      if (move != ttMove_) return move;
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
          if (m == countermove_) return countermove_;
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
      Move move = selectBestMove(quiets_, quietScores_, nextQuiet_, quietCount_);
      
      // Skip moves already returned
      if (move == ttMove_ || move == countermove_) continue;
      bool isKiller = false;
      for (int i = 0; i < killerCount_; ++i) {
        if (move == killerMoves_[i]) {
          isKiller = true;
          break;
        }
      }
      if (isKiller) continue;

    return move;
    }
    phase_ = 6;
  }

  // Phase 6: Bad Captures (Deferred from Phase 2)
  if (phase_ == 6) {
    while (nextBadCapture_ < badCaptureCount_) {
      Move move = badCaptures_[nextBadCapture_++];
      // Skip TT move (though it should have been good anyway)
      if (move != ttMove_) return move;
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
template<size_t N>
Move MovePicker::selectBestMove(std::array<Move, N> &list,
                                std::array<int, N> &scores, int &nextIdx, int count) {
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

// Piece values for SEE (in centipawns)
static constexpr int SEE_PIECE_VALUES[7] = {
    100,   // PAWN
    320,   // KNIGHT
    330,   // BISHOP
    500,   // ROOK
    900,   // QUEEN
    20000, // KING
    0      // NONE
};

namespace {
    uint64_t pieceSquareHash[12][64];
    bool hashesInitialized = false;
    void initPieceSquareHashes() {
        if (hashesInitialized) return;
        uint64_t seed = 0x9E3779B97F4A7C15ULL;
        for (int i=0; i<12; i++) {
            for (int sq=0; sq<64; sq++) {
                seed ^= seed >> 12;
                seed ^= seed << 25;
                seed ^= seed >> 27;
                pieceSquareHash[i][sq] = seed * 2685821657736338717ULL;
            }
        }
        hashesInitialized = true;
    }

    double LMRTable[SearchConstants::MAX_PLY + 1][256];
    bool lmrInitialized = false;
    void initLMR() {
        if (lmrInitialized) return;
        for (int i = 0; i <= SearchConstants::MAX_PLY; i++) {
            for (int n = 0; n < 256; n++) {
                if (i > 0 && n > 0)
                    LMRTable[i][n] = 4.0 / 10.0 + std::log(i) * std::log(n) / (20.0 / 10.0);
                else
                    LMRTable[i][n] = 0;
            }
        }
        lmrInitialized = true;
    }
}

// Helper: check if a square is near the enemy king (within 2 squares)
static bool isNearKing(Square sq, Square kingSquare) {
  int fileDiff = std::abs(static_cast<int>(sq.file()) -
                          static_cast<int>(kingSquare.file()));
  int rankDiff = std::abs(static_cast<int>(sq.rank()) -
                          static_cast<int>(kingSquare.rank()));
    return fileDiff <= 2 && rankDiff <= 2;
}

// Helper: Normalize mate/TB scores for transposition table storage
// Converts from ply-relative to root-relative
static int value_to_tt(int value, int ply) {
  if (value >= SearchConstants::MATE_IN_MAX)
    return value + ply;
  if (value <= -SearchConstants::MATE_IN_MAX)
    return value - ply;
  // Handle Tablebase Scores (similar to mate)
  if (value >= SearchConstants::TB_WIN_IN_MAX)
    return value + ply;
  if (value <= -SearchConstants::TB_WIN_IN_MAX)
    return value - ply;
    return value;
}

// Helper: Denormalize mate/TB scores when retrieving from transposition table
// Converts from root-relative back to ply-relative
static int value_from_tt(int value, int ply) {
  if (value >= SearchConstants::MATE_IN_MAX)
    return value - ply;
  if (value <= -SearchConstants::MATE_IN_MAX)
    return value + ply;
  // Handle Tablebase Scores (similar to mate)
  if (value >= SearchConstants::TB_WIN_IN_MAX)
    return value - ply;
  if (value <= -SearchConstants::TB_WIN_IN_MAX)
    return value + ply;
    return value;
}

Negamax::Negamax(IEvaluator &eval, TranspositionTable &table,
                 std::shared_ptr<SearchSharedData> shared, bool isMainThread,
                 int threadId)
    : evalCtx_(eval), tt_(table), shared_(shared), isMainThread_(isMainThread),
      threadId_(threadId) {
  h_ = std::make_unique<SearchHeuristics>();
  
  initPieceSquareHashes();
  initLMR();
  clearState();
}

void Negamax::clearState() {
  for (auto &row : h_->killers_) {
    row[0] = Move(0);
    row[1] = Move(0);
  }
  
  h_->failHighCount_.fill(0);
  
  // History and Continuation History persist across moves for better warm-up
  // They use internal exponential gravity (entry += bonus - entry * abs(bonus) / 16384)
  // which naturally manages decay/stale data.

  // Clear killers and fail-high counts only
  // We don't clear history_ or contHist_ or captureHist_ here.
  
  prevMove_.fill(Move(0)); // Clear previous move tracking
  prevPiece_.fill(Piece::NONE);
  
  // No longer clearing contHist_ or captureHist_ for persistence

  for (auto &row : h_->pvTable_) {
    row.fill(Move(0));
  }
  pvLength_.fill(0);

  // Decay correction history instead of clearing
  // Gravity-based decay: halve all entries to fade stale corrections
  for (auto& side : h_->pawnCorrHist_) {
    for (auto& v : side) v /= 2;
  }
  for (auto& side : h_->nonPawnCorrHist_) {
    for (auto& color : side) {
      for (auto& v : color) v /= 2;
    }
  }
  for (auto& d1 : h_->contCorrHist_) {
    for (auto& d2 : d1) {
      for (auto& d3 : d2) {
        for (auto& v : d3) v /= 2;
      }
    }
  }
  
  evalHistory_.fill(SearchConstants::INVALID_EVAL);

  // Reset time management state
  lastBestMove_ = Move(0);
  lastScore_ = 0;
  bestMoveChanges_ = 0;
  scoreDrops_ = 0;
  rootNodeCounts_.fill(0);
  searchNodes_ = 0;
}

Move Negamax::startSearch(Board &root, const SearchParams &params) {
  rootNodeCounts_.fill(0);
  searchNodes_ = 0;
  
  // Only Main Thread handles initialization of shared state and time
  if (isMainThread_) {
    shared_->stop = false;
    shared_->totalNodes = 0;
    shared_->startTime = std::chrono::steady_clock::now();
    localNodes_ = 0; // Reset local counter

    // Store searchMoves for root filtering
    searchMoves_ = params.searchMoves;

    // Calculate Time Limits using TimeManager
    // Get game phase from FeatureExtractor (need to extract features)
    ChessInput features{};
    FeatureExtractor::fill_input(root, features);
    float phase = features.global[FeatureExtractor::GlobalIndices::PHASE];
    int moveNum = root.fullMoveNumber();

    timeAlloc_ = timeManager_.calculate(params, root.sideToMove(), phase,
                                        lastEvalCp_, moveNum);
    syncTimeToShared();

    std::cout << "info string time soft=" << timeAlloc_.softLimit
              << "ms hard=" << timeAlloc_.hardLimit
              << "ms optimal=" << timeAlloc_.optimalTime << "ms"
              << " phase=" << phase << std::endl;


    // Start new search (age TT entries)
    // IMPORTANT: Always increment age, even for ponder searches!
    tt_.newSearch();

    // CHEKMATE/STALEMATE CHECK AT ROOT
    // If the root position has no legal moves, we must return immediately.
    // Otherwise the search loop will run with 0 moves, returning -INFINITE score,
    // which results in "mate --49999".
    Movelist rootMoves;
    MoveGen::generateLegalMoves(rootMoves, root);
    
    if (rootMoves.empty()) {
        int score = root.inCheck() ? -SearchConstants::MATE_SCORE : SearchConstants::DRAW_SCORE;
        
        // Report result
        if (infoCallback_) {
            // Send depth 0 info with precise score
            infoCallback_(0, score, 0, 0, {});
        }
        
        // Ensure we stop
        shared_->stop = true;
        
    return Move(0);
    }
  }

  // All threads clear their local history/killers for a fresh search
  // Note: In Lazy SMP, retaining history might be beneficial, but for now we
  // clear
  clearState();

  // --- Tablebase Probe (Main Thread Only, Skip During Ponder) ---
  // Don't probe TB during ponder - we can't use opponent's TB move anyway,
  // and the output can leak into the next search causing wrong moves.
  if (isMainThread_ && !params.ponder && Tablebase::available(root)) {
    auto tbResult = Tablebase::probeRoot(root);
    if (tbResult) {
      Move tbMove = std::get<0>(*tbResult);
      Tablebase::WDL wdl = std::get<1>(*tbResult);
      int dtz = std::get<2>(*tbResult);

      // PROMOTION FIX: DTZ doesn't distinguish mate speed.
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
            int eval = evalCtx_.evaluate(testBoard);
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

        if (infoCallback_) {
          std::vector<Move> pv = {tbMove};
          // Use WDL + DTZ for score
          int tbScore;
          if (wdl == Tablebase::WDL::WIN) {
            tbScore = TB_WIN_SCORE - (dtz * 2);
          } else if (wdl == Tablebase::WDL::LOSS) {
            tbScore = -TB_WIN_SCORE + (dtz * 2);
          } else if (wdl == Tablebase::WDL::CURSED_WIN) {
            tbScore = 100;
          } else if (wdl == Tablebase::WDL::BLESSED_LOSS) {
            tbScore = -100;
          } else {
            tbScore = DRAW_SCORE;
          }
          infoCallback_(0, tbScore, 0, 0, pv);
        }

        // Stop other threads immediately
        shared_->stop = true;
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
        std::cout << "info string TB: probeRoot failed, using WDL fallback"
                  << std::endl;
        std::cout << "info string DEBUG: TB Fallback searching from FEN: "
                  << root.getFen() << std::endl;

        // Test each legal move and find one that maintains winning/losing
        // status
        Movelist moves;
        MoveGen::generateLegalMoves(moves, root);

        Move bestTbMove;
        int bestEval = -INFINITE;

        for (const Move &move : moves) {
          Board testBoard = root;
          testBoard.makeMove(move);

          auto childWdl = Tablebase::probeWDL(testBoard);
          if (!childWdl)
            continue;

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
            int eval = -evalCtx_.evaluate(testBoard);
            if (eval > bestEval) {
              bestEval = eval;
              bestTbMove = move;
            }
          }
        }

        if (bestTbMove != Move(0)) {
          // SAFETY CHECK: Verify move is legal before returning
          Piece piece = root.at(bestTbMove.from());
          if (piece == Piece::NONE || piece.color() != root.sideToMove()) {
            std::cout
                << "info string CRITICAL: TB Fallback selected illegal move "
                << chess::uci::moveToUci(bestTbMove)
                << " (Piece: " << (int)piece.type()
                << ", Color: " << (int)piece.color() << ")"
                << ". Falling back to search." << std::endl;
            // Do not return; fall through to normal search
          } else {
            std::cout << "info string TB: WDL fallback found "
                      << chess::uci::moveToUci(bestTbMove) << std::endl;

            int tbScore = Tablebase::wdlToScore(*wdl, 0);
            if (infoCallback_) {
              std::vector<Move> pv = {bestTbMove};
              infoCallback_(0, tbScore, 0, 0, pv);
            }

            shared_->stop = true;
    return bestTbMove;
          }
        }
      }
    }
  }

  Move bestMove = Move(Move::NO_MOVE);
  int bestScore = -INFINITE;
  int reachedDepth = 0;

  // Aspiration window parameters
  constexpr int ASPIRATION_WINDOW = 25;
  constexpr int HELPER_ASPIRATION_WINDOW = 50;  // Wider for helpers
  int alpha = -INFINITE;
  int beta = INFINITE;

  // Iterative Deepening
  int maxDepth = params.depth;

  // Store target depth for helper thread coordination
  if (isMainThread_) {
    shared_->targetDepth.store(maxDepth, std::memory_order_relaxed);
    shared_->mainThreadDepth.store(0, std::memory_order_relaxed);
    shared_->mainThreadScore.store(0, std::memory_order_relaxed);
  }

  // SMP Strategy: Helpers search same depths as main but with aspiration guidance
  // - Main thread: Full iterative deepening with aspiration windows
  // - Helper threads: Full depths but use main thread's score for windows
  //   This allows helpers to contribute meaningful TT entries at all depths
  //   while benefiting from main thread's search results
  int helperMaxDepth = maxDepth;
  if (!isMainThread_ && threadId_ > 0) {
    // Helpers search full depth now (no artificial limit)
    // They use main thread's score for aspiration and stop when main finishes
    helperMaxDepth = maxDepth;
  }

  uint64_t startHash = root.hash();
  chess::Color startStm = root.sideToMove();

  for (int depth = 1;
       depth <= (isMainThread_ ? maxDepth : helperMaxDepth); ++depth) {
    
    // VERIFICATION: Check board state at start of iteration
    if (root.hash() != startHash || root.sideToMove() != startStm) {
      std::cout << "info string CRITICAL: Board state corrupted after depth " << (depth - 1) 
                << "! StartHash: " << std::hex << startHash << " Now: " << root.hash() 
                << " StartStm: " << (int)startStm << " Now: " << (int)root.sideToMove() 
                << std::dec << std::endl;
      // Force recovery if possible, but search is likely compromised
    }
    currentIter_ = depth; // Ensure Singular Extensions activate correctly

    // Clear tree at start of each iteration (keeps only final depth data)
    if (isMainThread_ && shared_->exportTree) {
      shared_->vizTree.clear();
    }

    // =========================================================================
    // LAZY SMP FIX #3: Helper threads stop when main finishes target depth
    // =========================================================================
    if (!isMainThread_) {
      int mainDepth = shared_->mainThreadDepth.load(std::memory_order_relaxed);
      int targetDepth = shared_->targetDepth.load(std::memory_order_relaxed);
      
      // Stop if main thread has completed target depth
      if (mainDepth >= targetDepth && targetDepth > 0) {
        break;
      }
      
      // Also check stop flag
      if (shared_->stop.load(std::memory_order_relaxed)) {
        break;
      }
    }

    // =========================================================================
    // LAZY SMP FIX #2: Aspiration windows for helper threads
    // =========================================================================
    if (isMainThread_ && depth >= 4 && std::abs(bestScore) < MATE_IN_MAX) {
      // Main thread: use own previous score
      alpha = bestScore - ASPIRATION_WINDOW;
      beta = bestScore + ASPIRATION_WINDOW;
    } else if (!isMainThread_ && depth >= 3) {
      // Helper thread: use main thread's score with wider window
      int mainScore = shared_->mainThreadScore.load(std::memory_order_relaxed);
      int mainDepth = shared_->mainThreadDepth.load(std::memory_order_relaxed);
      
      // Use main thread's score if it's recent enough (within 2 depths)
      if (mainDepth >= depth - 2 && mainDepth > 0 && std::abs(mainScore) < MATE_IN_MAX) {
        // Use wider aspiration window for helpers (50cp instead of 25cp)
        // This allows some exploration while still benefiting from bounds
        int windowSize = HELPER_ASPIRATION_WINDOW + (depth - mainDepth) * 20;
        alpha = mainScore - windowSize;
        beta = mainScore + windowSize;
      } else {
        // Main thread too far behind - use full window
        alpha = -INFINITE;
        beta = INFINITE;
      }
    } else {
      alpha = -INFINITE;
      beta = INFINITE;
    }

    int score;
    int failCount = 0;

    // Aspiration window search with re-search on fail
    while (true) {
      score = alphaBeta(root, depth, alpha, beta, 0, true, 0, false);

      if (shared_->stop)
        break;

      // Fail low - widen alpha
      if (score <= alpha) {
        // Prevent infinite loop: if already at minimum, accept the score
        if (alpha <= -INFINITE + 1)
          break;
        alpha = std::max(alpha - (ASPIRATION_WINDOW << ++failCount), -INFINITE);
      }
      // Fail high - widen beta
      else if (score >= beta) {
        // Prevent infinite loop: if already at maximum, accept the score
        if (beta >= INFINITE - 1)
          break;
        beta = std::min(beta + (ASPIRATION_WINDOW << ++failCount), INFINITE);
      }
      // Score within window
      else {
        break;
      }

      // Fallback to full window after too many failures
      if (failCount >= 3) {
        alpha = -INFINITE;
        beta = INFINITE;
      }
    }

    // Check for stop signal
    if (shared_->stop)
      break;

    // Update best move from PV
    if (pvLength_[0] > 0) {
      Move newBestMove = h_->pvTable_[0][0];

      // Track best move changes for time management
      if (depth > 1 && lastBestMove_ != Move(0) &&
          newBestMove != lastBestMove_) {
        bestMoveChanges_++;
      }

      // Track significant score drops (> 30cp)
      if (depth > 1 && score < lastScore_ - 30) {
        scoreDrops_++;
      }

      lastBestMove_ = newBestMove;
      lastScore_ = score;
      bestMove = newBestMove;
      bestScore = score;
      
      // =========================================================================
      // LAZY SMP: Main thread publishes score and depth for helper threads
      // =========================================================================
      if (isMainThread_) {
        shared_->mainThreadScore.store(bestScore, std::memory_order_relaxed);
        shared_->mainThreadDepth.store(depth, std::memory_order_relaxed);
      }
    }

    // Only Main Thread reports info and checks soft time limits
    if (isMainThread_) {
      // Flush local node counter before reporting (for accurate display)
      if (localNodes_ > 0) {
        shared_->totalNodes.fetch_add(localNodes_, std::memory_order_relaxed);
        localNodes_ = 0;
      }

      reachedDepth = depth;

      if (infoCallback_) {
        std::vector<Move> pv(h_->pvTable_[0].begin(),
                             h_->pvTable_[0].begin() + pvLength_[0]);
        int64_t elapsed = elapsedMs();
        uint64_t totalNodes =
            shared_->totalNodes.load(std::memory_order_relaxed);
        int nps =
            elapsed > 0 ? static_cast<int>((totalNodes * 1000) / elapsed) : 0;
        infoCallback_(depth, bestScore, totalNodes, nps, pv);
      }

      // Dynamic time management decision
      // SAFETY: Never stop before minimum depth to avoid tactical blunders
      // Depth 3 is fast and catches most 1-move blunders like hanging pieces
      constexpr int MIN_SEARCH_DEPTH = 3;
      if (depth >= MIN_SEARCH_DEPTH &&
          shouldStopIteration(depth, bestScore, bestMove)) {
        shared_->stop = true;
        break;
      }

      // If we found a mate, no need to search deeper
      if (std::abs(bestScore) > MATE_IN_MAX)
        break;
      // Tablebase win (guaranteed win) – stop early as well
      if (bestScore >= TB_WIN_IN_MAX) {
        shared_->stop = true;
        break;
      }
    }
  }

  // MultiPV is only supported for Main Thread to avoid complexity
  // MultiPV mode: search for additional best moves (expensive - for analysis
  // only)
  if (isMainThread_ && params.multiPV > 1 && bestMove != Move(0) &&
      !shared_->stop) {
    // Output PV 1 (already found)
    if (pvLength_[0] > 0) {
      std::string pvStr;
      for (int j = 0; j < pvLength_[0]; ++j) {
        if (!pvStr.empty())
          pvStr += " ";
        pvStr += chess::uci::moveToUci(h_->pvTable_[0][j]);
      }
      std::cout << "info multipv 1 score cp " << (std::abs(bestScore) >= SearchConstants::MATE_IN_MAX ? bestScore : bestScore * 100 / 195) << " pv " << pvStr
                << std::endl;
    }

    // Search for additional PVs by excluding already-found moves
    std::vector<Move> excludedMoves;
    excludedMoves.push_back(bestMove);

    for (int pvNum = 2; pvNum <= params.multiPV && !shared_->stop; ++pvNum) {
      // Generate moves and filter out excluded ones
      Movelist allMoves;
      MoveGen::generateLegalMoves(allMoves, root);

      // Skip if no moves left to search
      if (static_cast<int>(allMoves.size()) <=
          static_cast<int>(excludedMoves.size()))
        break;

      // Search with excluded moves
      SearchParams subParams = params;
      subParams.multiPV = 1; // Disable recursive multiPV
      subParams.searchMoves.clear();

      // Add all non-excluded moves to searchMoves
      for (const Move &m : allMoves) {
        bool excluded = false;
        for (const Move &ex : excludedMoves) {
          if (m == ex) {
            excluded = true;
            break;
          }
        }
        if (!excluded) {
          subParams.searchMoves.push_back(m);
        }
      }

      if (subParams.searchMoves.empty())
        break;

      // Reset for sub-search
      clearState();
      shared_->stop = false; // Reset stop flag for sub-search

      // Disable time limits for MultiPV sub-searches (they should be quick
      // anyway)
      TimeManager::TimeAllocation savedAlloc = timeAlloc_;
      timeAlloc_.hardLimit = 0;
      timeAlloc_.softLimit = 0;

      // Update searchMoves_ for this sub-search
      searchMoves_ = subParams.searchMoves;

      // Quick search at final depth only (no full iterative deepening)
      int searchDepth =
          std::min(params.depth, 6); // Limit depth for secondary PVs

      int alpha = -INFINITE;
      int beta = INFINITE;
      int score = alphaBeta(root, searchDepth, alpha, beta, 0, true, 0, false);

      // Restore time limits
      timeAlloc_ = savedAlloc;

      if (shared_->stop)
        break;

      if (pvLength_[0] > 0) {
        Move subBestMove = h_->pvTable_[0][0];
        int subScore = score;

        std::string pvStr;
        for (int j = 0; j < pvLength_[0]; ++j) {
          if (!pvStr.empty())
            pvStr += " ";
          pvStr += chess::uci::moveToUci(h_->pvTable_[0][j]);
        }
        std::cout << "info multipv " << pvNum << " score cp " << (std::abs(subScore) >= SearchConstants::MATE_IN_MAX ? subScore : subScore * 100 / 195)
                  << " pv " << pvStr << std::endl;

        excludedMoves.push_back(subBestMove);
      } else {
        break;
      }
    }

    // Clear searchMoves after MultiPV
    searchMoves_.clear();
  }

  if (isMainThread_) {
    // Ensure everyone stops
    shared_->stop = true;

    // Debug logging
    if (tt_.isDebugMode()) {
      std::cout << "info string DEBUG: Search finished. Depth: " << reachedDepth
                << " Score: " << bestScore
                << " Move: " << chess::uci::moveToUci(bestMove)
                << " TT Util: " << tt_.hashfull() << "/1000" << std::endl;

      // Dump TT if requested via global flag (could be added to SearchParams)
      // or just log that we finished.
    }
  }
  // Safety Fallback: Ensure we never return an illegal/null move
  if (isMainThread_ &&
      (bestMove == Move(Move::NO_MOVE) || bestMove == Move(Move::NULL_MOVE))) {
    std::cout
        << "info string CRITICAL: Search returned no move. Attempting fallback."
        << std::endl;

    // 1. Try TT
    bestMove = tt_.probeMove(root.hash());

    // 2. If valid move isn't found, pick *any* legal move
    Movelist legalMoves;
    MoveGen::generateLegalMoves(legalMoves, root);

    bool isLegal = false;
    if (bestMove != Move(Move::NO_MOVE)) {
      for (const auto &m : legalMoves) {
        if (m == bestMove) {
          isLegal = true;
          break;
        }
      }
    }

    if (!isLegal) {
      if (legalMoves.size() > 0) {
        bestMove = legalMoves[0];
        std::cout << "info string WARNING: Using fallback legal move "
                  << chess::uci::moveToUci(bestMove) << std::endl;
      } else {
        // Stalemate/Checkmate but search didn't catch it? Should be handled
        // upstream. If we are here, it's bad.
        std::cout << "info string FATAL: No legal moves available in fallback."
                  << std::endl;
      }
    }
  }

    return bestMove;
}

// Helper for Lazy Eval Pruning - The "Security Guard"
bool Negamax::canLazySkip(const Board &board, int depth, int alpha, int beta,
                          int ply, int fastScore, Bound &outBound,
                          bool prevWasCapture, bool pvNode) {

  // Check if feature is enabled
  if (!shared_->enableLazyEval)
    return false;

  // Never lazy on PV or near root
  if (pvNode)
    return false;

  if (ply < 2)
    return false;

  // Depth limit
  if (depth > shared_->lazyEvalMaxDepth)
    return false;

  // Tactical safety
  if (board.inCheck())
    return false;

  if (prevWasCapture)
    return false;

  // Check prevMove for promotions (encoded in move)
  if (ply > 0) {
    Move lastMove = prevMove_[ply - 1];
    if (lastMove != Move(0) && lastMove.typeOf() == Move::PROMOTION) {
    return false;
    }
  }

  // Avoid mate-range using the new helper
  if (isMateScore(fastScore))
    return false;

  // Margin Calculation
  // We use a base margin + depth scaling.
  // Additionally, we increase margin for "unclear" positions (score near 0)
  // where variance is highest.
  int margin =
      shared_->lazyEvalBaseMargin + (depth * shared_->lazyEvalDepthMargin);

  // Score-Dependent Safety: Increase margin if score is close to 0
  // If abs(score) < 500, add up to 100cp extra safety.
  // This ensures we rarely cut off "potential" in balanced positions.
  int scoreMagnitude = std::abs(fastScore);
  if (scoreMagnitude < 500) {
    margin += (500 - scoreMagnitude) / 5;
  }

  margin = std::max(margin, shared_->lazyEvalMinMargin);

  // Fail-High: We are winning by so much that NN won't bring us down to Beta
  if (fastScore >= beta + margin) {
    outBound = Bound::LOWER;
    return true;
  }

  // Fail-Low: We are losing by so much that NN won't save us above Alpha

  // Special Case: If Alpha is super high (TB Win or Mate), "normal" material
  // scores will look like fail-lows (e.g. 2000 vs 800000). But we MUST NOT
  // prune, because a "normal" move might be a faster Mate! We only prune if we
  // are sure we can't beat Alpha. But Eval=2000 could become Mate=900000.
  if (alpha >= SearchConstants::TB_WIN_IN_MAX)
    return false;

  // SAFETY: Use a massive safety margin for Fail-Low to avoid pruning tactical
  // recoveries. Instead of generating captures (expensive), we just assume a
  // Queen swing (1000cp) is possible. We only prune if we are losing by (Margin
  // + 1000), meaning even a Queen capture won't save us.
  int failLowMargin = margin + 1000;

  if (fastScore <= alpha - failLowMargin) {
    outBound = Bound::UPPER;
    return true;
  }

    return false;
}

uint64_t Negamax::getPawnKey(const Board& board) const {
    uint64_t key = 0;
    chess::Bitboard pawns = board.pieces(chess::PieceType::PAWN);
    while (pawns) {
        chess::Square sq = pawns.pop();
        key ^= pieceSquareHash[static_cast<int>(board.at(sq).internal())][sq.index()];
    }
    return key;
}

uint64_t Negamax::getNonPawnKey(const Board& board, chess::Color c) const {
    uint64_t key = 0;
    chess::Bitboard np = board.us(c) & ~(board.pieces(chess::PieceType::PAWN) | board.pieces(chess::PieceType::KING));
    while (np) {
        chess::Square sq = np.pop();
        key ^= pieceSquareHash[static_cast<int>(board.at(sq).internal())][sq.index()];
    }
    return key;
}

int Negamax::alphaBeta(Board &board, int depth, int alpha, int beta, int ply,
                       bool allowNull, int extensions, bool prevWasCapture, bool cutnode, Move excludedMove) {
  int alphaOriginal = alpha;
  int betaOriginal = beta;
  // Safety check: prevent stack overflow from excessive recursion
  if (ply >= MAX_PLY - 1) {
    return evalCtx_.evaluate(board);
  }

  // Check for stop
  if (shouldStop()) {
    return 0;
  }

  // Initialize PV length
  pvLength_[ply] = ply;

  // Check for draw
  if (ply > 0) {
    if (isDraw(board)) {
    return DRAW_SCORE - CONTEMPT_FACTOR;
    }
  }

  // Leaf node: quiescence search
  if (depth <= 0) {
    return quiescence(board, alpha, beta, ply);
  }

  // Thread-local node counting with periodic flush for accurate time management
  localNodes_++;
  
  // === OPTIMIZATION: More frequent node sync (every 4096 nodes) ===
  // This ensures time checks use fresh node counts during long iterations
  if ((localNodes_ & 4095) == 0) {
    shared_->totalNodes.fetch_add(localNodes_, std::memory_order_relaxed);
    localNodes_ = 0;
  }

  bool inCheck = board.inCheck();
  bool pvNode = (beta - alpha) > 1;
  bool singularSearch = (excludedMove != Move(0));
  uint64_t posKey = board.hash();

  tt_.prefetch(posKey); // Prefetch TT entry to hide memory latency

  // Probe tablebase BEFORE TT to prevent cache pollution hiding guaranteed
  // wins TB results are exact and should override any cached search
  // inaccuracies Only probe when we have few pieces (already checked in
  // available) Allow probing at any depth (even ply 1) since TB lookups are
  // fast for 3/4/5 pieces and we want to stop search immediately.
  // State for Tablebase result

  if (ply > 0 && Tablebase::available(board)) {
    // Lazy Probing: Check WDL first (fast)
    auto wdl = Tablebase::probeWDL(board);
    if (wdl) {
      int lazyScore = Tablebase::wdlToScore(*wdl, ply);

      // 1. Fail High / Beta Cutoff (Winning or Draw refutation)
      // If the TB result is good enough to cause a beta cutoff, we don't need
      // the move. return immediately.
      if (lazyScore >= beta) {
        // Store Lower Bound or Exact?
        // TB Results are "Exact" in truth, but if we only probed WDL, we lack
        // accurate DTZ-based score modification for the "fastest win".
        // However, for a cutoff, the raw wdl score is sufficient.
        tt_.store(posKey, depth, value_to_tt(lazyScore, ply), Bound::LOWER,
                  Move(0), lazyScore);
    return lazyScore;
      }

      // 2. Fail Low / Alpha Cutoff
      // If the TB result is worse than alpha, we can't improve.
      if (lazyScore <= alpha) {
        tt_.store(posKey, depth, value_to_tt(lazyScore, ply), Bound::UPPER,
                  Move(0), lazyScore);
    return lazyScore;
      }

      // 3. PV Node or Window Search needing exact move
      // If we are here, alpha < lazyScore < beta.
      // We need the Best Move and accurate DTZ score to return EXACT.
      auto detailed = Tablebase::probeRoot(board);
      if (detailed) {
        int dtz = std::get<2>(*detailed);
        Move localTbMove = std::get<0>(*detailed);
        int exactScore;

        if (*wdl == Tablebase::WDL::WIN) {
          // Adjust score for DTZ (prefer shorter wins)
          exactScore = SearchConstants::TB_WIN_SCORE - (dtz + ply);
        } else if (*wdl == Tablebase::WDL::LOSS) {
          // Adjust score for DTZ (prefer longer losses)
          exactScore = -SearchConstants::TB_WIN_SCORE + (dtz + ply);
        } else {
          exactScore = lazyScore;
        }

        // Store Exact result
        tt_.store(posKey, depth, value_to_tt(exactScore, ply), Bound::EXACT,
                  localTbMove, exactScore);
    return exactScore;
      } else {
        // Fallback (should rarely happen if WDL worked)
    return lazyScore;
      }
    }
  }

  // Lockless TT Probe transposition table
  Move ttMove = Move(0);
  int ttScore;
  int ttDepth = -1;
  Bound foundBound = Bound::NONE;
  // posKey already defined above
  
  // Bug #2 Fix: If singularSearch is true, force ttHit = false
  // This avoids re-searching the ttMove/excludedMove through the TT lookup
  bool ttHit = false;
  if (!singularSearch) {
    ttHit = tt_.probe(posKey, depth, alpha, beta, ttScore, ttMove, ttDepth,
                       foundBound);
  }

  if (ttHit && !pvNode) {
    // PV FIX: Even on TT cutoff, we should at least provide the TT move as a 1-move PV
    // This prevents the parent from seeing stale/garbage PV data from other branches.
    if (ttMove != Move(0)) {
      h_->pvTable_[ply][ply] = ttMove;
      pvLength_[ply] = ply + 1;
    }
    return value_from_tt(ttScore, ply);
  }

  // Internal Iterative Reduction
  // Apply when PV or cutnode has no TT move at sufficient depth
  // Only when NOT in check - never reduce depth during evasions
  if ((pvNode || cutnode) && ttMove == Move(0) && depth > 2 && !singularSearch && !inCheck) {
    depth--;
  }

  // Static evaluation - try TT cache first (fast path)
  int eval;
  int rawEval; // Uncorrected eval for TT storage
  int16_t cachedScore;
  bool evalComputed = false; // Flag to track if we actually computed NN eval
  bool ttEvalHit = tt_.probeEval(posKey, cachedScore);

  // Gate Pruning: Use TT bounds to skip NN eval in internal nodes
  bool canSkipEval = false;

  if (ttEvalHit) {
    // 1. If we have an EXACT TT score, use it as the static eval.
    // Even if it's from a shallower search, a search score is a better proxy
    // than a raw static eval (depth 0).
    if (foundBound == Bound::EXACT) {
      eval = value_from_tt(ttScore, ply);
      rawEval = eval;  // Approximate - TT score is already refined
      canSkipEval = true;
    }
    // 2. TT has cached raw eval - apply current corrhist correction
    else {
      rawEval = cachedScore;
      // Re-apply current corrections to the raw eval
      int pc = static_cast<int>(board.sideToMove());
      uint64_t pKey = getPawnKey(board) % 16384;
      uint64_t npKeyW = getNonPawnKey(board, chess::Color::WHITE) % 16384;
      uint64_t npKeyB = getNonPawnKey(board, chess::Color::BLACK) % 16384;
      int correction = h_->pawnCorrHist_[pc][pKey] + 
                       h_->nonPawnCorrHist_[pc][0][npKeyW] + 
                       h_->nonPawnCorrHist_[pc][1][npKeyB];

      if (ply >= 2 && prevMove_[ply - 1] != Move(0) && prevMove_[ply - 2] != Move(0)
          && prevPiece_[ply - 1] != Piece::NONE && prevPiece_[ply - 2] != Piece::NONE) {
          int pPiece2 = static_cast<int>(prevPiece_[ply - 2]);
          int pTo2 = prevMove_[ply - 2].to().index();
          int pPiece1 = static_cast<int>(prevPiece_[ply - 1]);
          correction += h_->contCorrHist_[pPiece2][pTo2][pPiece1][pTo2];
      }

      eval = rawEval * (200 - board.halfMoveClock()) / 200;
      if (shared_->config.enableCorrHist) {
        eval = std::clamp(eval + shared_->config.corrWeight * correction / 512, -SearchConstants::MATE_SCORE + SearchConstants::MAX_PLY, SearchConstants::MATE_SCORE - SearchConstants::MAX_PLY);
      }
      canSkipEval = true;
    }
  }

  // --- NN Evaluation (no lazy skip) ---
  if (!canSkipEval) {
    rawEval = evalCtx_.evaluate(board);
    
    // --- Eval Correction ---
    int pc = static_cast<int>(board.sideToMove());
    uint64_t pKey = getPawnKey(board) % 16384;
    uint64_t npKeyW = getNonPawnKey(board, chess::Color::WHITE) % 16384;
    uint64_t npKeyB = getNonPawnKey(board, chess::Color::BLACK) % 16384;

    int correction = h_->pawnCorrHist_[pc][pKey] + 
                     h_->nonPawnCorrHist_[pc][0][npKeyW] + 
                     h_->nonPawnCorrHist_[pc][1][npKeyB];

    if (ply >= 2 && prevMove_[ply - 1] != Move(0) && prevMove_[ply - 2] != Move(0)
        && prevPiece_[ply - 1] != Piece::NONE && prevPiece_[ply - 2] != Piece::NONE) {
        int pPiece2 = static_cast<int>(prevPiece_[ply - 2]);
        int pTo2 = prevMove_[ply - 2].to().index();
        int pPiece1 = static_cast<int>(prevPiece_[ply - 1]);
        correction += h_->contCorrHist_[pPiece2][pTo2][pPiece1][pTo2];
    }
                     
    eval = rawEval * (200 - board.halfMoveClock()) / 200;
    if (shared_->config.enableCorrHist) {
      eval = std::clamp(eval + shared_->config.corrWeight * correction / 512, -SearchConstants::MATE_SCORE + SearchConstants::MAX_PLY, SearchConstants::MATE_SCORE - SearchConstants::MAX_PLY);
    }

    evalComputed = true; // Mark as fresh NN eval
    // Store RAW eval in TT (not corrected!) so future visits can re-correct
    tt_.storeEval(posKey, static_cast<int16_t>(rawEval));
  }
  
  // Record eval history for improving heuristic
  evalHistory_[ply] = evalComputed ? eval : (canSkipEval ? eval : SearchConstants::INVALID_EVAL);

  // --------------------------------------------------------------

  // Razoring: at low depths, if eval is way below alpha, drop to qsearch
  if (!pvNode && !inCheck && !singularSearch && alpha < 2000 && depth < 5 &&
      eval + 400 * depth < alpha) {
    int razorScore = quiescence(board, alpha, beta, ply);
    if (razorScore <= alpha) {
        return razorScore;
    }
  }

  // Reverse Futility Pruning
  // Compute improving: is our eval better than 2 plies ago (or 4 if 2 unavailable)?
  bool improving = false;
  if (!inCheck && ply >= 2) {
    int pastEval = evalHistory_[ply - 2];
    if (pastEval == SearchConstants::INVALID_EVAL && ply >= 4) {
      pastEval = evalHistory_[ply - 4];
    }
    if (pastEval != SearchConstants::INVALID_EVAL) {
      improving = eval > pastEval;
    }
  }

  if (!pvNode && !inCheck && !singularSearch && depth <= 8 &&
      eval - shared_->config.reverseFutilityMargin * (depth - improving) >= beta &&
      std::abs(beta) < MATE_IN_MAX &&
      std::abs(eval) < MATE_IN_MAX) { // Tightened guard
    return (eval + beta) / 2;
  }

  // Null Move Pruning
  if (!singularSearch && allowNull && !inCheck && !pvNode && depth >= 3 && eval >= beta &&
      board.hasNonPawnMaterial(board.sideToMove())) {
    board.makeNullMove();
    prevMove_[ply] = Move(0);
    prevPiece_[ply] = Piece::NONE;

    // Dynamic reduction
    int R = 4 + depth / 5 + std::min(3, (eval - beta) / 175);

    int nullScore = -alphaBeta(board, depth - 1 - R, -beta, -beta + 1, ply + 1,
                               false, extensions, false, !cutnode);
    board.unmakeNullMove();

    if (shared_->stop)
      return 0;

    if (nullScore >= beta) {
      if (nullScore >= MATE_IN_MAX)
        nullScore = beta;
      return nullScore;
    }
  }

  // ProbCut
  // If we are doing well, and a capture beats a high beta margin on a shallow search,
  // we can safely assume the full search will also fail high and prune the node.
  if (!singularSearch && !inCheck && depth >= 5 && std::abs(beta) < MATE_IN_MAX - MAX_PLY) {
      int pBeta = beta + 250; // ProbCut Margin
      bool doProbCut = true;
      
      // If TT move indicates low score, we skip ProbCut
      if (ttHit && ttMove != Move(0) && ttDepth >= depth - 3 && ttScore < pBeta) {
          doProbCut = false;
      }
      
      if (doProbCut) {
          Movelist probcutMoves;
          MoveGen::generateCaptures(probcutMoves, board);
          int threshold = pBeta - eval;
          
          for (const Move& move : probcutMoves) {
              if (see(board, move) >= threshold) {
                  board.makeMove(move);
                  
                  int pcScore = -quiescence(board, -pBeta, -pBeta + 1, ply + 1);
                  if (pcScore >= pBeta) {
                      pcScore = -alphaBeta(board, depth - 4, -pBeta, -pBeta + 1, ply + 1, false, extensions, true);
                  }
                  
                  board.unmakeMove(move);
                  
                  if (pcScore >= pBeta) {
                      return pcScore;
                  }
              }
          }
      }
  }

  // --------------------------------------------------------------
  // Move Generation and Ordering
  // --------------------------------------------------------------

  // Clear the fail-high counter for the grandchildren
  if (ply + 2 < SearchConstants::MAX_PLY) {
    h_->failHighCount_[ply + 2] = 0;
  }

  // Generate and score moves
  Movelist moves;
  MoveGen::generateLegalMoves(moves, board);

  // Filter root moves if searchMoves is specified (for MultiPV and UCI
  // searchmoves)
  if (ply == 0 && !searchMoves_.empty()) {
    Movelist filteredMoves;
    for (const Move &m : moves) {
      for (const Move &sm : searchMoves_) {
        if (m == sm) {
          filteredMoves.add(m);
          break;
        }
      }
    }
    moves = filteredMoves;
  }

  // Checkmate or stalemate
  if (moves.size() == 0) {
    return inCheck ? -mateScore(ply) : DRAW_SCORE;
  }

  // The TT might return a move that is illegal in this position (hash
  // collision). Using an invalid move will cause crashes when accessing
  // squares or making moves.
  if (ttMove != Move(0)) {
    bool isValid = false;
    for (const auto &m : moves) {
      if (m == ttMove) {
        isValid = true;
        break;
      }
    }
    if (!isValid) {
      ttMove = Move(0); // Discard corrupted/invalid TT move
    }
  }

  // =========================================================================
  // LAZY SMP FIX #4: Improved thread diversity
  // =========================================================================
  // Old approach: Skip TT move at 25% of nodes (every 4th ply) - too aggressive
  // New approach: Skip TT move ONLY at root for deep searches, and less often
  // This preserves the value of TT hits while still providing exploration
  if (!isMainThread_ && threadId_ > 0 && ttMove != Move(0)) {
    if (ply == 0 && depth >= 6) {
      // Root diversification: Each thread prefers different root moves
      // Skip TT move based on thread ID and move count
      int rootMoveCount = static_cast<int>(moves.size());
      if (rootMoveCount > 1) {
        // Thread N skips TT move every (numThreads) searches at root
        // This ensures each thread explores different first moves occasionally
        int skipFrequency = std::max(2, shared_->numThreads);
        if ((threadId_ % skipFrequency) == 0) {
          ttMove = Move(0);  // Force exploration of other root moves
        }
      }
    } else if (ply > 0 && depth >= 8) {
      // Deep nodes only: Skip TT move much less often (every 8th ply, not 4th)
      // This reduces duplicate work while preserving good move ordering
      if ((ply + threadId_) % 8 == 0) {
        ttMove = Move(0);
      }
    }
  }

  // Create move picker with thread ID for diversified move ordering
  // Look up countermove: the move that previously refuted opponent's last move
  Move countermove;
  if (ply > 0) {
    Move prevMove = prevMove_[ply - 1];
    if (prevMove != Move(0)) {
      countermove =
          h_->countermove_[prevMove.from().index()][prevMove.to().index()];
    }
  }
  MovePicker picker(moves, ttMove, &h_->killers_, &h_->history_, &h_->captureHist_, ply, threadId_,
                    countermove);

  // Futility Pruning
  int lmrDepth = depth;
  
  bool canFutilityPrune =
      !pvNode && !inCheck && depth < 7 &&
      eval + 98 + 121 * lmrDepth < alpha &&
      std::abs(alpha) < MATE_IN_MAX;

  // Late Move Pruning threshold
  int lmpThreshold = shared_->config.lmpBase + depth * depth / (2 - improving);

  Move bestMove = Move(0);
  int bestScore = -INFINITE;
  Bound ttFlag = Bound::UPPER;
  int movesSearched = 0;
  Move searchedQuiets[256];
  int numSearchedQuiets = 0;

  // Use MovePicker instead of iterating all moves
  Move move;
  while ((move = picker.nextMove(board, this, inCheck)) != Move(0)) {
    
    if (singularSearch && move == excludedMove) continue;

    uint64_t nodesBeforeMove = 0;
    if (ply == 0) {
      nodesBeforeMove = shared_->totalNodes.load(std::memory_order_relaxed) + localNodes_;
    }

    // Check move properties BEFORE making move
    bool isCapt = isCapture(board, move);
    bool isPromotion = move.typeOf() == Move::PROMOTION;
    bool isQuiet = !isCapt && !isPromotion;
    bool isTTMove = (move == ttMove);
    Piece movingPiece = board.at(move.from());

    // Get history score for history pruning
    int historyScore = h_->history_[move.from().index()][move.to().index()];

    if (!isCapt && !pvNode && bestScore > -MATE_IN_MAX) {
      int lmr_depth = std::max(1, depth - static_cast<int>(LMRTable[std::min(depth, SearchConstants::MAX_PLY)][std::min(movesSearched, 255)]));

      // Late Move Pruning
      if (depth < 6 && movesSearched >= lmpThreshold) {
        // Export pruned move
        if (isMainThread_ && shared_->exportTree && ply < shared_->exportTreeDepth) {
          using PR = SearchSharedData::PruneReason;
          shared_->vizTree.push_back({posKey, 0, chess::uci::moveToUci(move),
              INVALID_EVAL, 0, depth, ply, movesSearched, 0, false, PR::LMP});
        }
        continue;
      }

      // Futility Pruning
      if (canFutilityPrune && movesSearched > 0) {
        // Export pruned move
        if (isMainThread_ && shared_->exportTree && ply < shared_->exportTreeDepth) {
          using PR = SearchSharedData::PruneReason;
          shared_->vizTree.push_back({posKey, 0, chess::uci::moveToUci(move),
              INVALID_EVAL, 0, depth, ply, movesSearched, 0, false, PR::FUTILITY});
        }
        continue;
      }

      // History Pruning
      if (movesSearched > 0 && lmr_depth < 4 && historyScore < -4096 * lmr_depth) {
        continue;
      }
    }

    // SEE Pruning
    if (!pvNode && depth < 7 && movesSearched > 0 && !isPromotion &&
        (movingPiece.type() != PieceType::KING) && 
        bestScore > -MATE_IN_MAX &&
        std::abs(alpha) < MATE_IN_MAX) {
        
      int margin = isCapt ? -94 * depth : -86 * depth;
      
      if (see(board, move) < margin) {
        // Export pruned move
        if (isMainThread_ && shared_->exportTree && ply < shared_->exportTreeDepth) {
          using PR = SearchSharedData::PruneReason;
          shared_->vizTree.push_back({posKey, 0, chess::uci::moveToUci(move),
              INVALID_EVAL, 0, depth, ply, movesSearched, 0, false, PR::SEE});
        }
        continue;
      }
    }

    int extension = 0;
    
    // Advanced Singular Extensions
    if (ply > 0 && ply < currentIter_ * 2 && !singularSearch && depth >= 5 && move == ttMove && 
        foundBound != Bound::NONE && foundBound != Bound::UPPER &&
        std::abs(ttScore) < SearchConstants::MATE_SCORE - SearchConstants::MAX_PLY && 
        ttDepth >= depth - 3) {
        
        int ttScorePly = value_from_tt(ttScore, ply);
        int sBeta = ttScorePly - depth;
        int sScore = alphaBeta(board, (depth - 1) / 2, sBeta - 1, sBeta,
                               ply, false, extensions, prevWasCapture, cutnode, move);

        if (sScore < sBeta) {
            if (!pvNode && sScore + 18 < sBeta && ply < currentIter_) {
                extension = 2 + (!isCapt && sScore < sBeta - 126);
            } else {
                extension = 1;
            }
        } else if (sBeta >= beta) {
    return sBeta;
        } else if (cutnode) {
            extension = -1;
        }
    }

    // Bug #1 fix: Record moving piece BEFORE makeMove (after makeMove, from() is empty)
    Piece movingPieceCopy = board.at(move.from());
    board.makeMove(move);
    prevMove_[ply] = move; // Track move for child nodes (recapture/countermove)
    prevPiece_[ply] = movingPieceCopy; // Track piece for Continuation History

    // Track quiet moves for history malus
    if (isQuiet) {
      if (numSearchedQuiets < 64) {
        searchedQuiets[numSearchedQuiets++] = move;
      }
    }

    int score;
    bool givesCheck = board.inCheck();

    int newExtensions = extensions + extension;
    int newDepth = depth - 1 + extension;

    // Late Move Reductions
    // Minimal entry conditions, but MUST exclude in-check positions
    // Aggressive: enter LMR if movesSearched > 0 (or pvNode ? 0 : -1)

    if (depth >= LMR_MIN_DEPTH && !inCheck && movesSearched > 0) {

      // State-Aware Precomputed LMR
      int movesIdx = std::min(movesSearched, 255);
      int depthIdx = std::min(depth, SearchConstants::MAX_PLY);
      int reduction = static_cast<int>(LMRTable[depthIdx][movesIdx]);
      
      if (isCapt || isPromotion) {
          reduction /= 2;
      } else {
          int historyScore = h_->history_[move.from().index()][move.to().index()];
          reduction -= historyScore / 9818;
      }

      reduction -= (ttHit && ttDepth >= depth) ? 1 : 0;
      
      // Use the improving flag computed earlier
      
      reduction += !improving ? 1 : 0;
      reduction += cutnode ? 1 : 0;

      // Increase reduction if this branch is a tactical minefield
      if (ply + 1 < SearchConstants::MAX_PLY) {
        reduction += (h_->failHighCount_[ply + 1] > 4) ? 1 : 0;
      }

      // reduce less if move gives check
      reduction -= givesCheck ? 1 : 0;
      
      // SMP Diversification
      if (threadId_ > 0) {
        int lmrVariation = (threadId_ % 3) - 1; // -1, 0, or +1
        reduction += lmrVariation;
      }

      // Ensure reduction is reasonable (clamp logic)
      reduction = std::clamp(reduction, 0, newDepth - 1);

      // Reduced depth search with null window
      score = -alphaBeta(board, newDepth - reduction, -alpha - 1, -alpha, ply + 1, true, newExtensions, isCapt || isPromotion, true);

      // Re-search at full depth if reduced search improved alpha
      if (score > alpha) {
        score = -alphaBeta(board, newDepth, -alpha - 1, -alpha, ply + 1, true, newExtensions, isCapt || isPromotion, !cutnode);

        // Full window re-search for PV nodes if still beats alpha (important
        // for finding mates!)
        if (pvNode && score > alpha && score < beta) {
          score = -alphaBeta(board, newDepth, -beta, -alpha, ply + 1, true, newExtensions, isCapt || isPromotion, false);
        }
      }
    } else if (!pvNode || movesSearched > 0) {
      // Zero-window search for non-PV nodes
      // At root in tactical positions, use wider window to avoid missing
      // mates
      if (ply == 0 && alpha > 1000) {
        score = -alphaBeta(board, newDepth, -beta, -alpha, ply + 1, true,
                           newExtensions, isCapt || isPromotion, !cutnode);
      } else {
        score = -alphaBeta(board, newDepth, -alpha - 1, -alpha, ply + 1, true, newExtensions, isCapt || isPromotion, !cutnode);
      }
    } else {
      score = alpha + 1;
    }

    if (pvNode && score > alpha && (movesSearched > 0 || score < beta)) {
      score = -alphaBeta(board, newDepth, -beta, -alpha, ply + 1, true, newExtensions, isCapt || isPromotion, false);
    }

    // --- Search Tree Export (Main Thread Only) ---
    if (isMainThread_ && shared_->exportTree) {
      bool inExportRange = (ply < shared_->exportTreeDepth);
      bool isPVMove = (score > alpha);
      
      if (inExportRange || isPVMove) {
        // Try to retrieve the child's static eval from the TT
        // It was computed and stored inside the child's alphaBeta call
        int16_t childEval;
        int exportStaticEval = 0;
        
        // Probe TT for the child position
        if (tt_.probeEval(board.hash(), childEval)) {
           // Negate because child's eval is from side-to-move (child's perspective)
           // We want it from parent's perspective
           exportStaticEval = -static_cast<int>(childEval);
        } else {
           // No eval found in TT for this child position
           // Use sentinel value to indicate "not evaluated"
           exportStaticEval = INVALID_EVAL; 
        }

        using PR = SearchSharedData::PruneReason;
        shared_->vizTree.push_back({
            posKey,              // Parent Hash
            board.hash(),        // Child Hash (current board before unmake)
            chess::uci::moveToUci(move),
            exportStaticEval,    // Static eval of CHILD position (or sentinel)
            score,               // Search score (backed up)
            depth,               // Remaining depth
            ply,                 // Ply from root
            movesSearched,       // Move order (0-indexed, before increment)
            0,                   // subtreeNodes - TODO: track per-move
            isPVMove,            // Is this on PV?
            score >= beta ? PR::BETA_CUTOFF : PR::NONE
        });
      }
    }

    board.unmakeMove(move);
    
    if (ply == 0) {
      uint64_t nodesAfterMove = shared_->totalNodes.load(std::memory_order_relaxed) + localNodes_;
      uint16_t moveIndex = (move.from().index() << 6) | move.to().index();
      rootNodeCounts_[moveIndex] += (nodesAfterMove - nodesBeforeMove);
      searchNodes_ += (nodesAfterMove - nodesBeforeMove);
    }

    movesSearched++;

    if (shared_->stop)
      return 0;

    // Print root evaluation scores only when UCI debug mode is tracked on
    if (ply == 0 && tt_.isDebugMode()) {
      std::cout << "info string ROOT EVAL move=" << chess::uci::moveToUci(move) << " score=" << score << std::endl;
    }

    if (score > bestScore) {
      bestScore = score;
      bestMove = move;

      if (score > alpha) {
        alpha = score;
        ttFlag = Bound::EXACT;

        // Update PV
        h_->pvTable_[ply][ply] = move;
        int childPVLength = pvLength_[ply + 1];
        // Ensure we don't copy more than MAX_PLY
        int copyEnd = std::min(childPVLength, SearchConstants::MAX_PLY);
        for (int i = ply + 1; i < copyEnd; ++i) {
          h_->pvTable_[ply][i] = h_->pvTable_[ply + 1][i];
        }
        // Correctly set length: if child had no PV, length is ply + 1
        pvLength_[ply] = std::max(ply + 1, childPVLength);

        if (score >= beta) {
          ttFlag = Bound::LOWER;

          // Record that the current ply caused a fail-high
          if (ply < SearchConstants::MAX_PLY) {
            h_->failHighCount_[ply]++;
          }

          // Update killers and history for quiet moves
          if (isQuiet) {
            updateKillers(move, ply);
            
            int bonus = std::min(
                291 * (depth - 1 + (score > beta + 125 ? 1 : 0)), 2476);
            
            updateHistory(board, move, bonus, ply);

            // History Malus: penalize all other searched quiets
            for (int i = 0; i < numSearchedQuiets; i++) {
              if (searchedQuiets[i] != move) {
                int malus = -bonus * 15 / (10 + std::min(i, 30));
                updateHistory(board, searchedQuiets[i], malus, ply);
              }
            }

            // Update countermove table
            if (ply > 0) {
              Move prevMove = prevMove_[ply - 1];
              if (prevMove != Move(0)) {
                int prevFrom = prevMove.from().index();
                int prevTo = prevMove.to().index();
                h_->countermove_[prevFrom][prevTo] = move;
              }
            }
          } else {
            // Update Capture History
            // Grab attacker type from movingPiece (captured before makeMove)
            int attackerType = static_cast<int>(movingPiece.type());
            int toSq = move.to().index();
            int bonus = std::min(291 * (depth - 1), 2476);
            int& entry = h_->captureHist_[attackerType][toSq];
            entry = entry + bonus - entry * std::abs(bonus) / 16384;
          }
          break;
        }
      }
    }
  }

  // --- Eval Correction History Update ---
  // FIX: Use the CORRECTED 'eval' for conditions and bonus, NOT the raw 'staticEvalToStore'
  if (shared_->config.enableCorrHist && !inCheck && (bestMove == Move(0) || !isCapture(board, bestMove)) &&
      !(bestScore >= betaOriginal && bestScore <= eval) &&
      !(bestMove == Move(0) && bestScore >= eval)) {
      
      // Ensure eval is valid and we are not in mate scores
      if (eval != SearchConstants::INVALID_EVAL && 
          bestScore > -SearchConstants::MATE_SCORE + SearchConstants::MAX_PLY && 
          bestScore < SearchConstants::MATE_SCORE - SearchConstants::MAX_PLY) {
          
          // CALCULATE BONUS USING CORRECTED EVAL
          int bonus = std::clamp((bestScore - eval) * depth / 8, -256, 256);
          
          int pc = static_cast<int>(board.sideToMove());
          uint64_t pKey = getPawnKey(board) % 16384;
          uint64_t npKeyW = getNonPawnKey(board, chess::Color::WHITE) % 16384;
          uint64_t npKeyB = getNonPawnKey(board, chess::Color::BLACK) % 16384;
          
          auto update_corr = [](int16_t& entry, int b) {
              entry = entry + b - entry * std::abs(b) / 1024;
          };
          update_corr(h_->pawnCorrHist_[pc][pKey], bonus);
          update_corr(h_->nonPawnCorrHist_[pc][0][npKeyW], bonus);
          update_corr(h_->nonPawnCorrHist_[pc][1][npKeyB], bonus);

          if (ply >= 2 && prevMove_[ply - 1] != Move(0) && prevMove_[ply - 2] != Move(0)
              && prevPiece_[ply - 1] != Piece::NONE && prevPiece_[ply - 2] != Piece::NONE) {
              int pPiece2 = static_cast<int>(prevPiece_[ply - 2]);
              int pTo2 = prevMove_[ply - 2].to().index();
              int pPiece1 = static_cast<int>(prevPiece_[ply - 1]);
              update_corr(h_->contCorrHist_[pPiece2][pTo2][pPiece1][pTo2], bonus);
          }
      }
  }

  // Determine the raw eval to store in the TT
  int staticEvalToStore = SearchConstants::INVALID_EVAL;
  if (evalComputed) {
    staticEvalToStore = rawEval; // TT still gets the raw eval!
  } else if (ttEvalHit) {
    staticEvalToStore = cachedScore; 
  }

  if (!singularSearch) {
    tt_.store(posKey, depth, value_to_tt(bestScore, ply), ttFlag, bestMove,
              staticEvalToStore);
  }

  return bestScore;
}

int Negamax::quiescence(Board &board, int alpha, int beta, int ply) {
  // Time check in quiescence
  if ((localNodes_ & 1023) == 0) {
    if (shouldStop())
      return 0;
  }

  localNodes_++;

  // Depth limit to prevent infinite recursion
  constexpr int MAX_QSEARCH_PLY = 24;
  if (ply >= MAX_PLY - 1 || ply >= MAX_QSEARCH_PLY) {
    int16_t cachedScore;
    uint64_t posKey = board.hash();
    if (tt_.probeEval(posKey, cachedScore)) {
      return cachedScore;
    }
    int e = evalCtx_.evaluate(board);
    tt_.storeEval(posKey, static_cast<int16_t>(e));
    return e;
  }

  bool inCheck = board.inCheck();
  uint64_t posKey = board.hash();

  // Full TT probe with bound checking
  Move ttMove = Move(0);
  int ttScore;
  int ttDepth = -1;
  Bound foundBound = Bound::NONE;
  bool ttHit = tt_.probe(posKey, 0, alpha, beta, ttScore, ttMove, ttDepth,
                         foundBound);

  int rawEval = SearchConstants::INVALID_EVAL;
  int ttStaticEval = SearchConstants::INVALID_EVAL;

  if (ttHit) {
    ttScore = value_from_tt(ttScore, ply);
    // TT cutoff with bound checking
    if (ttScore != SearchConstants::INVALID_EVAL) {
      if ((foundBound == Bound::EXACT) ||
          (foundBound == Bound::LOWER && ttScore >= beta) ||
          (foundBound == Bound::UPPER && ttScore <= alpha)) {
        return ttScore;
      }
    }
  }

  int bestScore = -SearchConstants::INFINITE;
  Move bestMove = Move(0);
  bool raisedAlpha = false;

  int staticEval = SearchConstants::INVALID_EVAL;

  if (!inCheck) {
    // Compute or retrieve raw eval
    int16_t cachedScore;
    if (tt_.probeEval(posKey, cachedScore)) {
      rawEval = cachedScore;
    } else {
      rawEval = evalCtx_.evaluate(board);
      tt_.storeEval(posKey, static_cast<int16_t>(rawEval));
    }

    // Apply correction history
    // FIX: Must manually cast halfMoveClock to int, otherwise the entire expression
    // evaluates as unsigned, converting negative rawEvals into 4.29B unsigned ints 
    // which divided by 200 gives the cursed 21474221 values!
    staticEval = rawEval * (200 - static_cast<int>(board.halfMoveClock())) / 200;
    if (shared_->config.enableCorrHist) {
      int pc = static_cast<int>(board.sideToMove());
      uint64_t pKey = getPawnKey(board) % 16384;
      uint64_t npKeyW = getNonPawnKey(board, chess::Color::WHITE) % 16384;
      uint64_t npKeyB = getNonPawnKey(board, chess::Color::BLACK) % 16384;
      int correction = h_->pawnCorrHist_[pc][pKey] +
                       h_->nonPawnCorrHist_[pc][0][npKeyW] +
                       h_->nonPawnCorrHist_[pc][1][npKeyB];
      staticEval = std::clamp(staticEval + shared_->config.corrWeight * correction / 512,
                              -SearchConstants::MATE_SCORE + SearchConstants::MAX_PLY,
                              SearchConstants::MATE_SCORE - SearchConstants::MAX_PLY);
    }

    bestScore = staticEval;

    // Use TT score to refine stand-pat
    if (ttHit && ttScore != SearchConstants::INVALID_EVAL) {
      if (foundBound == Bound::EXACT ||
          (foundBound == Bound::UPPER && ttScore < staticEval) ||
          (foundBound == Bound::LOWER && ttScore > staticEval)) {
        bestScore = ttScore;
      }
    }

    if (bestScore >= beta) {
      return bestScore;
    }
    if (bestScore > alpha) {
      alpha = bestScore;
    }
  }

  // Generate moves
  Movelist moves;
  if (inCheck) {
    MoveGen::generateLegalMoves(moves, board);
    if (moves.size() == 0) {
      return -mateScore(ply);
    }
    if (ply >= MAX_QSEARCH_PLY) {
      if (rawEval != SearchConstants::INVALID_EVAL) return rawEval;
      int e = evalCtx_.evaluate(board);
      tt_.storeEval(posKey, static_cast<int16_t>(e));
      return e;
    }
  } else {
    MoveGen::generateCaptures(moves, board);
  }

  // MVV-LVA scoring
  for (auto &m : moves) {
    int score = 0;
    if (m.typeOf() == Move::ENPASSANT) {
      score = static_cast<int>(PieceType::PAWN) * 100;
    } else {
      Piece captured = board.at(m.to());
      if (captured != Piece::NONE) {
        score = static_cast<int>(captured.type()) * 100;
      }
    }
    if (m.typeOf() == Move::PROMOTION) {
      score += 500;
    }
    m.setScore(static_cast<int16_t>(score));
  }

  std::sort(moves.begin(), moves.end(),
            [](const Move &a, const Move &b) { return a.score() > b.score(); });

  int movesPlayed = 0;

  for (const Move &move : moves) {
    // Skip non-captures when not in check
    if (!inCheck && board.at(move.to()) == Piece::NONE && 
        move.typeOf() != Move::ENPASSANT && move.typeOf() != Move::PROMOTION) {
      continue;
    }

    // Delta pruning
    if (!inCheck && staticEval != SearchConstants::INVALID_EVAL && move.typeOf() != Move::PROMOTION) {
      Piece captured = board.at(move.to());
      int captureValue = 0;
      if (captured != Piece::NONE) {
        captureValue = SEE_PIECE_VALUES[static_cast<int>(captured.type())];
      } else if (move.typeOf() == Move::ENPASSANT) {
        captureValue = SEE_PIECE_VALUES[0]; // Pawn
      }
      if (staticEval + captureValue + 200 < alpha) {
        continue;
      }
    }

    // SEE filtering
    if (!inCheck && move.typeOf() != Move::ENPASSANT &&
        see(board, move) < -107) {
      continue;  // Skip bad captures
    }
    
    movesPlayed++;


    board.makeMove(move);
    int score = -quiescence(board, -beta, -alpha, ply + 1);
    board.unmakeMove(move);

    if (shared_->stop)
    return bestScore > -SearchConstants::INFINITE ? bestScore : 0;

    if (score > bestScore) {
      bestScore = score;
      if (score > alpha) {
        bestMove = move;
        raisedAlpha = true;
        alpha = score;
      }
      if (score >= beta) {
        break;
      }
    }
  }

  // If in check and no legal moves were played, it's checkmate
  if (inCheck && movesPlayed == 0) {
    return -mateScore(ply);
  }

  // If not in check and no captures were searched, return stand-pat
  if (movesPlayed == 0) {
    return bestScore;
  }

  // Store TT entry
  // TT::store() clamps scores to int16_t range automatically
  Bound entryType = bestScore >= beta ? Bound::LOWER
                  : raisedAlpha       ? Bound::EXACT
                                      : Bound::UPPER;

  tt_.store(posKey, 0, value_to_tt(bestScore, ply), entryType, bestMove,
            rawEval);

  return bestScore;
}

// scoreMoves and orderMoves have been replaced by MovePicker (see above)
// MovePicker uses lazy selection instead of upfront sorting

void Negamax::updateKillers(Move move, int ply) {
  if (move != h_->killers_[ply][0]) {
    h_->killers_[ply][1] = h_->killers_[ply][0];
    h_->killers_[ply][0] = move;
  }
}

void Negamax::updateHistory(const Board& board, Move move, int bonus, int ply) {
  int fromSq = move.from().index();
  int toSq = move.to().index();
  int piece = static_cast<int>(board.at(move.from()));

  // Exponential gravity update
  int entry = h_->history_[fromSq][toSq];
  h_->history_[fromSq][toSq] = entry + bonus - entry * std::abs(bonus) / 16384;
  
  // Continuation History update
  if (ply >= 1) {
    Move prev1 = prevMove_[ply - 1];
    if (prev1 != Move(0)) {
      int pPiece1 = static_cast<int>(prevPiece_[ply - 1]);
      int pTo1 = prev1.to().index();
      int& ch1 = h_->contHist_[pPiece1][pTo1][piece][toSq];
      ch1 = ch1 + bonus - ch1 * std::abs(bonus) / 16384;
    }
  }
  if (ply >= 2) {
    Move prev2 = prevMove_[ply - 2];
    if (prev2 != Move(0)) {
      int pPiece2 = static_cast<int>(prevPiece_[ply - 2]);
      int pTo2 = prev2.to().index();
      int& ch2 = h_->contHist_[pPiece2][pTo2][piece][toSq];
      ch2 = ch2 + bonus - ch2 * std::abs(bonus) / 16384;
    }
  }
  if (ply >= 4) {
    Move prev4 = prevMove_[ply - 4];
    if (prev4 != Move(0)) {
      int pPiece4 = static_cast<int>(prevPiece_[ply - 4]);
      int pTo4 = prev4.to().index();
      int b4 = bonus / 2;
      int& ch4 = h_->contHist_[pPiece4][pTo4][piece][toSq];
      ch4 = ch4 + b4 - ch4 * std::abs(b4) / 16384;
    }
  }
}

// Sync TimeManager allocation to shared state
void Negamax::syncTimeToShared() {
  if (shared_) {
    shared_->softLimit = timeAlloc_.softLimit;
    shared_->hardLimit = timeAlloc_.hardLimit;
  }
}

bool Negamax::shouldStopIteration(int depth, int score, Move bestMove) const {
  // Create allocation from shared limits (supports dynamic ponderhit updates)
  // IMPORTANT: Read from shared_ to support ponderhit time updates
  TimeManager::TimeAllocation currentAlloc = timeAlloc_;
  currentAlloc.softLimit = shared_->softLimit;
  currentAlloc.hardLimit = shared_->hardLimit;
  
  // Phase 2:
  // Compares nodes spent on the best move vs total nodes to scale time allocation
  if (searchNodes_ > 0 && bestMove != Move(0)) {
    uint16_t moveIndex = (bestMove.from().index() << 6) | bestMove.to().index();
    uint64_t bestMoveNodes = rootNodeCounts_[moveIndex];
    double fract = static_cast<double>(bestMoveNodes) / static_cast<double>(searchNodes_);
    
    // NodeTmFactors and bmStability
    int bmStability = std::max(0, depth - bestMoveChanges_);
    double factor = (1.81 - fract) * 2.15;
    double bmFactor = 1.27 - (bmStability * 0.06);
    
    // Scale soft limit dynamically
    int64_t scaledSoft = static_cast<int64_t>(currentAlloc.softLimit * factor * bmFactor);
    
    // Don't let it drop below 10% of the original soft limit to avoid skipping critical depths entirely
    currentAlloc.softLimit = std::max(currentAlloc.softLimit / 10, std::min(scaledSoft, currentAlloc.hardLimit));
  }
  
  // Delegate to TimeManager
  return timeManager_.shouldStop(elapsedMs(), depth, bestMoveChanges_,
                                 scoreDrops_, currentAlloc);
}

bool Negamax::shouldStop() const {
  if (shared_->stop.load(std::memory_order_relaxed))
    return true;

  // Helper threads only check the flag
  if (!isMainThread_)
    return false;

  // Main thread checks hard limit every 128 nodes
  // CRITICAL: Use localNodes_ for frequency, not shared counter!
  // shared_->totalNodes is only flushed at end of each depth, so during
  // a long iteration it would never trigger time checks.
  //
  // IMPORTANT: Read hardLimit from shared_ (not timeAlloc_) to support
  // dynamic time updates from ponderhit. When ponderhit arrives, UCI
  // updates shared_->hardLimit and shared_->startTime, and the search
  // will pick up the new limits here.
  int64_t hardLimit = shared_->hardLimit;
  if ((localNodes_ & 127) == 0 && hardLimit > 0) {
    int64_t elapsed = elapsedMs();
    // Hard limit check - stop when exceeded
    if (elapsed >= hardLimit) {
      shared_->stop = true;
    return true;
    }
  }
    return false;
}

int64_t Negamax::elapsedMs() const {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             now - shared_->startTime)
      .count();
}

bool Negamax::isCapture(const Board &board, Move move) const {
    return board.at(move.to()) != Piece::NONE || move.typeOf() == Move::ENPASSANT;
}

bool Negamax::isDraw(const Board &board) const {
  // 50-move rule
  if (board.halfMoveClock() >= 100)
    return true;

  // Threefold repetition: position must have occurred at least twice before
  if (board.isRepetition(2))
    return true;

  // Insufficient material is handled by chess.hpp if needed
    return false;
}

int Negamax::mateScore(int ply) const { return MATE_SCORE - ply; }

// Static Exchange Evaluation (SEE) - Swap Algorithm
// Properly handles batteries, X-rays, and overloaded defenders
// Returns the material balance after a series of optimal captures on a square
//
// Algorithm: Iteratively identify the least valuable attacker from each side,
// swap it out, and recalculate attacks with updated occupancy. This handles
// X-rays automatically as pieces move/capture reveal attackers behind them.
//
// Time Complexity: O(number of pieces attacking the square) - typically 4-8
// iterations
int Negamax::see(const Board &board, Move move) const {
  Square toSq = move.to();
  Square fromSq = move.from();

  int gain[32];
  Piece capturedPiece = board.at(toSq);

  // 1. Calculate the initial capture value
  if (move.typeOf() == Move::ENPASSANT) {
    gain[0] = SEE_PIECE_VALUES[0]; // Pawn Value
  } else {
    gain[0] = (capturedPiece != Piece::NONE)
                  ? SEE_PIECE_VALUES[static_cast<int>(capturedPiece.type())]
                  : 0;
  }

  // Quick exit for non-captures (optional, but good for performance)
  if (gain[0] == 0 && move.typeOf() != Move::ENPASSANT) {
      return 0;
  }

  Bitboard occupied = board.occ();
  occupied ^= Bitboard::fromSquare(fromSq);
  occupied ^= Bitboard::fromSquare(toSq);

  if (move.typeOf() == Move::ENPASSANT) {
    Square capturedPawnSq = Square(toSq.index() ^ 8);
    occupied ^= Bitboard::fromSquare(capturedPawnSq);
  }

  // 2. CRITICAL FIX: The piece standing on toSq for the NEXT capture is the
  //    piece that just moved (the attacker), NOT the captured piece.
  PieceType pieceOnToSq;
  if (move.typeOf() == Move::PROMOTION) {
    pieceOnToSq = move.promotionType();
  } else {
    pieceOnToSq = board.at(fromSq).type();
  }

  Color us = board.sideToMove();
  Color sideToMove = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
  int depth = 0;

  while (depth < 31) {
    depth++;

    // Find attackers (same as your corrected code)
    // Ensure you intersect with & occupied
    Bitboard attackingPawns =
        (chess::attacks::pawn(
             sideToMove == Color::WHITE ? Color::BLACK : Color::WHITE, toSq) &
         board.pieces(PieceType::PAWN, sideToMove) & occupied);
    Bitboard attackingKnights =
        (chess::attacks::knight(toSq) &
         board.pieces(PieceType::KNIGHT, sideToMove) & occupied);
    Bitboard attackingBishops =
        (chess::attacks::bishop(toSq, occupied) &
         board.pieces(PieceType::BISHOP, sideToMove) & occupied);
    Bitboard attackingRooks =
        (chess::attacks::rook(toSq, occupied) &
         board.pieces(PieceType::ROOK, sideToMove) & occupied);
    Bitboard attackingQueens =
        ((chess::attacks::bishop(toSq, occupied) |
          chess::attacks::rook(toSq, occupied)) &
         board.pieces(PieceType::QUEEN, sideToMove) & occupied);
    Bitboard attackingKing =
        (chess::attacks::king(toSq) &
         board.pieces(PieceType::KING, sideToMove) & occupied);

    Bitboard allAttackers = attackingPawns | attackingKnights |
                            attackingBishops | attackingRooks |
                            attackingQueens | attackingKing;

    if (allAttackers.empty())
      break;

    // Select least valuable attacker
    PieceType attackerType;
    Bitboard attacker;

    if (!attackingPawns.empty()) {
      attacker = Bitboard::fromSquare(attackingPawns.lsb());
      attackerType = PieceType::PAWN;
    } else if (!attackingKnights.empty()) {
      attacker = Bitboard::fromSquare(attackingKnights.lsb());
      attackerType = PieceType::KNIGHT;
    } else if (!attackingBishops.empty()) {
      attacker = Bitboard::fromSquare(attackingBishops.lsb());
      attackerType = PieceType::BISHOP;
    } else if (!attackingRooks.empty()) {
      attacker = Bitboard::fromSquare(attackingRooks.lsb());
      attackerType = PieceType::ROOK;
    } else if (!attackingQueens.empty()) {
      attacker = Bitboard::fromSquare(attackingQueens.lsb());
      attackerType = PieceType::QUEEN;
    } else {
      attacker = attackingKing;
      attackerType = PieceType::KING;
    }

    // Calculate gain using the piece currently standing on the square
    gain[depth] =
        SEE_PIECE_VALUES[static_cast<int>(pieceOnToSq)] - gain[depth - 1];

    occupied ^= attacker;
    sideToMove = (sideToMove == Color::WHITE) ? Color::BLACK : Color::WHITE;

    // Update pieceOnToSq for the NEXT iteration
    pieceOnToSq = attackerType;

    if (attackerType == PieceType::KING)
      break;
  }

  while (depth > 0) {
    gain[depth - 1] = -std::max(0, gain[depth]);
    depth--;
  }
    return gain[0];
}

int Negamax::pieceValue(PieceType pt) const {
    return SEE_PIECE_VALUES[static_cast<int>(pt)];
}

// Check if a pawn is a passed pawn
bool Negamax::isPassedPawn(const Board &board, Square sq, Color side) const {
  int file = static_cast<int>(sq.file());
  int rank = static_cast<int>(sq.rank());

  // Check files: same file and adjacent files for enemy pawns
  for (int f = std::max(0, file - 1); f <= std::min(7, file + 1); ++f) {
    // Check ranks in front of the pawn
    if (side == Color::WHITE) {
      for (int r = rank + 1; r <= 7; ++r) {
        Square checkSq = Square(static_cast<File>(f), static_cast<Rank>(r));
        Piece p = board.at(checkSq);
        if (p != Piece::NONE && p.type() == PieceType::PAWN &&
            p.color() == Color::BLACK) {
    return false;
        }
      }
    } else {
      for (int r = rank - 1; r >= 0; --r) {
        Square checkSq = Square(static_cast<File>(f), static_cast<Rank>(r));
        Piece p = board.at(checkSq);
        if (p != Piece::NONE && p.type() == PieceType::PAWN &&
            p.color() == Color::WHITE) {
    return false;
        }
      }
    }
  }

    return true;
}

// Check if move is a pawn push to the 7th rank (pre-promotion)
bool Negamax::isPawnPush7th(const Board &board, Move move) const {
  Piece p = board.at(move.from());
  if (p.type() != PieceType::PAWN)
    return false;

  int toRank = static_cast<int>(move.to().rank());
  Color side = p.color();

  // 7th rank is index 6 for white, index 1 for black
  return (side == Color::WHITE && toRank == 6) ||
         (side == Color::BLACK && toRank == 1);
}
