/**
 * @file Tablebase.cpp
 * @brief Syzygy Endgame Tablebase probe interface.
 *
 * Wraps the Fathom library to probe Syzygy tablebases (WDL and DTZ) for perfect 
 * play in endgames with a limited number of pieces.
 * @ingroup engine
 */
#include "Tablebase.h"
extern "C" {
#include "../../external/fathom/src/tbprobe.h"
}
#include <cstring>
#include <iostream>

namespace Tablebase {

static bool initialized_ = false;
static int maxPieces_ = 0;

bool init(const std::string &path) {
  if (path.empty()) {
    std::cout << "info string TB: no path specified" << std::endl;
    return false;
  }

  // Initialize Fathom
  bool success = tb_init(path.c_str());

  if (success) {
    maxPieces_ = TB_LARGEST;
    initialized_ = true;
    std::cout << "info string TB: initialized with " << maxPieces_
              << " piece tables from " << path << std::endl;
  } else {
    std::cout << "info string TB: failed to initialize from " << path
              << std::endl;
    initialized_ = false;
    maxPieces_ = 0;
  }

  return success;
}

bool available(const Board &board) {
  if (!initialized_ || maxPieces_ == 0)
    return false;

  int pieceCount = board.occ().count();
  return pieceCount <= maxPieces_;
}

std::optional<WDL> probeWDL(const Board &board) {
  if (!available(board))
    return std::nullopt;

  // Get bitboards - Fathom expects combined bitboards for both colors
  uint64_t white = board.us(Color::WHITE).getBits();
  uint64_t black = board.us(Color::BLACK).getBits();

  // Combined piece bitboards (both colors)
  uint64_t kings = board.pieces(PieceType::KING).getBits();
  uint64_t queens = board.pieces(PieceType::QUEEN).getBits();
  uint64_t rooks = board.pieces(PieceType::ROOK).getBits();
  uint64_t bishops = board.pieces(PieceType::BISHOP).getBits();
  uint64_t knights = board.pieces(PieceType::KNIGHT).getBits();
  uint64_t pawns = board.pieces(PieceType::PAWN).getBits();

  // Fathom's tb_probe_wdl requires rule50=0 and castling=0
  // It returns the theoretical WDL value ignoring 50-move rule.
  unsigned rule50 = 0;
  unsigned castling = 0;
  unsigned ep = (board.enpassantSq() != Square::underlying::NO_SQ)
                    ? board.enpassantSq().index()
                    : 0;
  bool turn = (board.sideToMove() == Color::WHITE);

  unsigned result = tb_probe_wdl(white, black, kings, queens, rooks, bishops,
                                 knights, pawns, rule50, castling, ep, turn);

  if (result == TB_RESULT_FAILED) {
    return std::nullopt;
  }

  // Convert Fathom WDL to our enum
  switch (result) {
  case TB_WIN:
    return WDL::WIN;
  case TB_CURSED_WIN:
    return WDL::CURSED_WIN;
  case TB_DRAW:
    return WDL::DRAW;
  case TB_BLESSED_LOSS:
    return WDL::BLESSED_LOSS;
  case TB_LOSS:
    return WDL::LOSS;
  default:
    return std::nullopt;
  }
}

std::optional<std::tuple<Move, WDL, int>> probeRoot(const Board &board) {
  if (!available(board))
    return std::nullopt;

  // Get bitboards - Fathom expects combined bitboards for both colors
  uint64_t white = board.us(Color::WHITE).getBits();
  uint64_t black = board.us(Color::BLACK).getBits();

  // Combined piece bitboards (both colors)
  uint64_t kings = board.pieces(PieceType::KING).getBits();
  uint64_t queens = board.pieces(PieceType::QUEEN).getBits();
  uint64_t rooks = board.pieces(PieceType::ROOK).getBits();
  uint64_t bishops = board.pieces(PieceType::BISHOP).getBits();
  uint64_t knights = board.pieces(PieceType::KNIGHT).getBits();
  uint64_t pawns = board.pieces(PieceType::PAWN).getBits();

  // Fathom requires rule50=0 for reliable probing if exact 50mr handling is not
  // guaranteed
  unsigned rule50 = 0;
  unsigned ep = (board.enpassantSq() != Square::underlying::NO_SQ)
                    ? board.enpassantSq().index()
                    : 0;
  bool turn = (board.sideToMove() == Color::WHITE);

  // Get castling rights
  // NOTE: Tablebase positions should NEVER have castling rights available.
  unsigned castling = 0;

  // Results array for probe
  unsigned results[TB_MAX_MOVES];

  unsigned result =
      tb_probe_root(white, black, kings, queens, rooks, bishops, knights, pawns,
                    rule50, castling, ep, turn, results);

  if (result == TB_RESULT_FAILED) {
    return std::nullopt;
  }

  // Extract move, WDL, and DTZ
  unsigned from = TB_GET_FROM(result);
  unsigned to = TB_GET_TO(result);
  unsigned promotes = TB_GET_PROMOTES(result);
  unsigned dtz = TB_GET_DTZ(result);
  unsigned wdl_raw = TB_GET_WDL(result);

  // Convert to chess.hpp Move
  Square fromSq = Square(static_cast<Square::underlying>(from));
  Square toSq = Square(static_cast<Square::underlying>(to));

  Move bestMove;
  if (promotes) {
    // Promotion move
    PieceType promoteTo;
    switch (promotes) {
    case TB_PROMOTES_QUEEN:
      promoteTo = PieceType::QUEEN;
      break;
    case TB_PROMOTES_ROOK:
      promoteTo = PieceType::ROOK;
      break;
    case TB_PROMOTES_BISHOP:
      promoteTo = PieceType::BISHOP;
      break;
    case TB_PROMOTES_KNIGHT:
      promoteTo = PieceType::KNIGHT;
      break;
    default:
      promoteTo = PieceType::QUEEN;
    }
    bestMove = Move::make<Move::PROMOTION>(fromSq, toSq, promoteTo);
  } else {
    // Check for castling
    Piece movingPiece = board.at(fromSq);
    if (movingPiece.type() == PieceType::KING &&
        std::abs(static_cast<int>(fromSq.file()) -
                 static_cast<int>(toSq.file())) > 1) {
      // Castling move
      bestMove = Move::make<Move::CASTLING>(fromSq, toSq);
    } else {
      // Normal or en passant
      bestMove = Move::make(fromSq, toSq);
    }
  }

  // Convert WDL to our enum
  WDL wdl_result;
  switch (wdl_raw) {
  case TB_WIN:
    wdl_result = WDL::WIN;
    break;
  case TB_CURSED_WIN:
    wdl_result = WDL::CURSED_WIN;
    break;
  case TB_DRAW:
    wdl_result = WDL::DRAW;
    break;
  case TB_BLESSED_LOSS:
    wdl_result = WDL::BLESSED_LOSS;
    break;
  case TB_LOSS:
    wdl_result = WDL::LOSS;
    break;
  default:
    wdl_result = WDL::DRAW;
  }

  return std::make_tuple(bestMove, wdl_result, static_cast<int>(dtz));
}

int wdlToScore(WDL wdl, int mate_ply) {
  using namespace SearchConstants;

  // Use a score slightly below MATE_SCORE for TB wins
  // This distinguishes "Winning Endgame" from "Actual Checkmate"
  // Score range: [TB_WIN_IN_MAX+1 ... TB_WIN_SCORE] is reserved for real mates
  // TB Wins: TB_WIN_IN_MAX (30200)
  switch (wdl) {
  case WDL::WIN:
    // Return TB_WIN_SCORE - ply
    return TB_WIN_SCORE - mate_ply;
  case WDL::CURSED_WIN:
    return 1; // Slight advantage
  case WDL::DRAW:
    return DRAW_SCORE;
  case WDL::BLESSED_LOSS:
    return -1; // Slight disadvantage
  case WDL::LOSS:
    return -TB_WIN_SCORE + mate_ply;
  }
  return DRAW_SCORE;
}

int maxPieces() { return maxPieces_; }

bool isInitialized() { return initialized_; }

} // namespace Tablebase
