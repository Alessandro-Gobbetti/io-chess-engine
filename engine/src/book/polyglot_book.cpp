#include "polyglot_book.hpp"
#include "polyglot_keys.hpp"
#include <algorithm>
#include <bit>
#include <fstream>
#include <iostream>

// Helper for endian swap
#if defined(_MSC_VER)
#include <stdlib.h>
#define bswap_64(x) _byteswap_uint64(x)
#define bswap_16(x) _byteswap_ushort(x)
#elif defined(__GNUC__) || defined(__clang__)
#define bswap_64(x) __builtin_bswap64(x)
#define bswap_16(x) __builtin_bswap16(x)
#else
// Generic fallback
uint64_t bswap_64(uint64_t val) {
  val = ((val << 8) & 0xFF00FF00FF00FF00ULL) |
        ((val >> 8) & 0x00FF00FF00FF00FFULL);
  val = ((val << 16) & 0xFFFF0000FFFF0000ULL) |
        ((val >> 16) & 0x0000FFFF0000FFFFULL);
  return (val << 32) | (val >> 32);
}
uint16_t bswap_16(uint16_t val) { return (val << 8) | (val >> 8); }
#endif

PolyglotBook::PolyglotBook(const std::string &book_path)
    : book_path_(book_path) {}

PolyglotBook::~PolyglotBook() {}

bool PolyglotBook::load() {
  std::ifstream stream(book_path_, std::ios::binary | std::ios::ate);
  if (!stream) {
    std::cerr << "info string Could not open book file: " << book_path_
              << std::endl;
    return false;
  }

  std::streamsize size = stream.tellg();
  stream.seekg(0, std::ios::beg);

  if (size % sizeof(PolyglotEntry) != 0) {
    std::cerr << "info string Book file has invalid size." << std::endl;
    return false;
  }

  entries_.resize(size / sizeof(PolyglotEntry));
  if (!stream.read(reinterpret_cast<char *>(entries_.data()), size)) {
    std::cerr << "info string Failed to read book file." << std::endl;
    entries_.clear();
    return false;
  }

  std::cout << "info string Loaded opening book with " << entries_.size()
            << " entries." << std::endl;
  return true;
}

uint64_t PolyglotBook::compute_polyglot_key(const chess::Board &board) const {
  uint64_t key = 0;
  using namespace chess;

  // Polyglot piece encoding: for each piece type, black first (offset 64*2*pt),
  // then white (offset 64*(2*pt+1))
  PieceType piece_types[] = {PieceType::PAWN,   PieceType::KNIGHT,
                             PieceType::BISHOP, PieceType::ROOK,
                             PieceType::QUEEN,  PieceType::KING};

  for (int pt = 0; pt < 6; ++pt) {
    PieceType p_type = piece_types[pt];

    // Black pieces: offset = 64 * (2 * pt)
    Bitboard b_pieces = board.pieces(p_type, Color::BLACK);
    while (b_pieces) {
      Square sq = b_pieces.pop();
      int offset = 64 * (2 * pt) + sq.index();
      key ^= Polyglot::Random64[offset];
    }

    // White pieces: offset = 64 * (2 * pt + 1)
    Bitboard w_pieces = board.pieces(p_type, Color::WHITE);
    while (w_pieces) {
      Square sq = w_pieces.pop();
      int offset = 64 * (2 * pt + 1) + sq.index();
      key ^= Polyglot::Random64[offset];
    }
  }

  const auto &cr = board.castlingRights();
  if (cr.has(Color::WHITE, Board::CastlingRights::Side::KING_SIDE))
    key ^= Polyglot::Random64[768];
  if (cr.has(Color::WHITE, Board::CastlingRights::Side::QUEEN_SIDE))
    key ^= Polyglot::Random64[768 + 1];
  if (cr.has(Color::BLACK, Board::CastlingRights::Side::KING_SIDE))
    key ^= Polyglot::Random64[768 + 2];
  if (cr.has(Color::BLACK, Board::CastlingRights::Side::QUEEN_SIDE))
    key ^= Polyglot::Random64[768 + 3];

  if (board.enpassantSq() != Square::NO_SQ) {
    Square ep_sq = board.enpassantSq();
    File ep_file = ep_sq.file();

    Color us = board.sideToMove();
    Color them = ~us;
    Bitboard enemy_pawns = board.pieces(PieceType::PAWN, them);

    Bitboard attackers = attacks::pawn(us, ep_sq) & enemy_pawns;

    if (attackers) {
      key ^= Polyglot::Random64[772 + static_cast<int>(ep_file)];
    }
  }

  // Turn: XOR if WHITE is to move
  if (board.sideToMove() == Color::WHITE) {
    key ^= Polyglot::Random64[780];
  }

  return key;
}

chess::Move PolyglotBook::polyglot_to_move(uint16_t poly_move,
                                           const chess::Board &board) const {
  using namespace chess;
  int to_file = (poly_move >> 0) & 7;
  int to_rank = (poly_move >> 3) & 7;
  int from_file = (poly_move >> 6) & 7;
  int from_rank = (poly_move >> 9) & 7;
  int prom_idx = (poly_move >> 12) & 7;

  Square from = Square(from_rank * 8 + from_file);
  Square to = Square(to_rank * 8 + to_file);

  PieceType prom = PieceType::NONE;
  if (prom_idx > 0) {
    switch (prom_idx) {
    case 1:
      prom = PieceType::KNIGHT;
      break;
    case 2:
      prom = PieceType::BISHOP;
      break;
    case 3:
      prom = PieceType::ROOK;
      break;
    case 4:
      prom = PieceType::QUEEN;
      break;
    }
  }

  // The chess.hpp library is a bit picky about move creation.
  // We must find the exact legal move that matches.
  Movelist moves;
  movegen::legalmoves(moves, board);
  for (const Move &move : moves) {
    if (move.from() == from && move.to() == to) {
      if (move.typeOf() == Move::PROMOTION) {
        if (move.promotionType() == prom) {
          return move;
        }
      } else {
        if (prom == PieceType::NONE) {
          return move;
        }
      }
    }
  }

  return Move::NULL_MOVE;
}

chess::Move PolyglotBook::probe(const chess::Board &board,
                                const std::string &bookName) {
  if (entries_.empty()) {
    return chess::Move::NULL_MOVE;
  }

  uint64_t key = compute_polyglot_key(board);

  auto it = std::lower_bound(entries_.begin(), entries_.end(), key,
                             [](const PolyglotEntry &entry, uint64_t k) {
                               return bswap_64(entry.key) < k;
                             });

  std::vector<PolyglotEntry> matches;
  while (it != entries_.end() && bswap_64(it->key) == key) {
    matches.push_back(*it);
    it++;
  }

  if (matches.empty()) {
    return chess::Move::NULL_MOVE;
  }

  // Print all book moves with weights
  std::cout << "info string " << bookName << " candidates:";
  for (const auto &entry : matches) {
    chess::Move move = polyglot_to_move(bswap_16(entry.move), board);
    std::string uci_str = chess::uci::moveToUci(move);
    uint16_t weight = bswap_16(entry.weight);
    std::cout << " " << uci_str << "(" << weight << ")";
  }
  std::cout << std::endl;

  // Weighted random selection
  uint32_t total_weight = 0;
  for (const auto &entry : matches) {
    total_weight += bswap_16(entry.weight);
  }

  if (total_weight == 0) { // Should not happen with good books
    chess::Move move = polyglot_to_move(bswap_16(matches[0].move), board);
    std::cout << "info string " << bookName
              << " selected: " << chess::uci::moveToUci(move)
              << " (only option)" << std::endl;
    return move;
  }

  uint32_t pick = rand() % total_weight;

  uint32_t current_sum = 0;
  for (const auto &entry : matches) {
    current_sum += bswap_16(entry.weight);
    if (current_sum > pick) {
      chess::Move move = polyglot_to_move(bswap_16(entry.move), board);
      std::cout << "info string " << bookName
                << " selected: " << chess::uci::moveToUci(move) << std::endl;
      return move;
    }
  }

  return chess::Move::NULL_MOVE;
}
