/**
 * @file FeatureExtractor.cpp
 * @brief Implementation of standard feature extraction.
 */
#include "FeatureExtractor.hpp"
#include <algorithm>
#include <cstring>

// --- Constants & Lookups ---
static constexpr uint8_t VAL_MAX = 255;
static constexpr uint8_t VAL_ON = 255;

struct alignas(64) ChebTable {
  uint8_t data[2][64][64];
};

static const auto CHEBYSHEV = []() {
  ChebTable t{};
  for (int sq1 = 0; sq1 < 64; ++sq1) {
    int r1 = sq1 / 8, c1 = sq1 % 8;

    // Table 0: White Perspective (Normal)
    for (int sq2 = 0; sq2 < 64; ++sq2) {
      int r2 = sq2 / 8, c2 = sq2 % 8;
      int dist = std::max(std::abs(r1 - r2), std::abs(c1 - c2));
      uint8_t val = static_cast<uint8_t>(dist * 36);
      t.data[0][sq1][sq2] = val;
    }

    // Table 1: Black Perspective (iterate output positions, compute which
    // square they map to)
    for (int out_sq = 0; out_sq < 64; ++out_sq) {
      int out_r = out_sq / 8;
      int out_c = out_sq % 8;
      // Unflip to get the normal square this output position corresponds to
      int sq2 = (7 - out_r) * 8 + out_c;
      int r2 = sq2 / 8, c2 = sq2 % 8;
      int dist = std::max(std::abs(r1 - r2), std::abs(c1 - c2));
      uint8_t val = static_cast<uint8_t>(dist * 36);
      t.data[1][sq1][out_sq] = val;
    }
  }
  return t;
}();

template <chess::PieceType::underlying PT> constexpr int get_index() {
  if constexpr (PT == chess::PieceType::PAWN)
    return 0;
  else if constexpr (PT == chess::PieceType::KNIGHT)
    return 1;
  else if constexpr (PT == chess::PieceType::BISHOP)
    return 2;
  else if constexpr (PT == chess::PieceType::ROOK)
    return 3;
  else if constexpr (PT == chess::PieceType::QUEEN)
    return 4;
  else if constexpr (PT == chess::PieceType::KING)
    return 5;
  else
    return 0; // Should never happen
}

// --- Feature Extraction ---
template <chess::Color::underlying US>
static void fill_input_impl(const chess::Board &board, ChessInput &out) {
  using enum FeatureExtractor::LayerIndices;
  using enum FeatureExtractor::GlobalIndices;

  // CLEAR MEMORY (Fastest way to init)
  // We only zero the sparse/dense parts. Globals are overwritten.
  std::memset(&out.layers, 0, sizeof(out.layers));

  // Perspective Setup
  constexpr chess::Color US_COLOR = static_cast<chess::Color>(US);
  constexpr chess::Color THEM_COLOR = ~US_COLOR;

  // "Orient": If it's Black's turn, flip indices vertically (Rank 1 <-> Rank
  // 8). If White: mask is 56 (00111000 binary), which flips the ranks. If
  // Black: mask is 0, which leaves the square unchanged.
  constexpr int ORIENT_XOR = (US_COLOR == chess::Color::WHITE) ? 56 : 0;
  constexpr int CHEB_IDX = (US_COLOR == chess::Color::WHITE) ? 1 : 0;

  // Helper: Write 255 to a specific plane
  auto set_sq = [&](int plane_idx, int sq) {
    out.layers[plane_idx][sq ^ ORIENT_XOR] = VAL_ON;
  };

  // Helper: Iterate Bitboard and fill
  auto fill_layer = [&](int plane_idx, chess::Bitboard bb) {
    while (bb) {
      set_sq(plane_idx, bb.pop()); // Uses optimized lsb/pop
    }
  };

  // =============================================================
  //  PIECES PRESENCE (0-11)
  // =============================================================

  auto fill_presence = [&]<chess::PieceType::underlying PT>() {
    constexpr int idx =
        get_index<PT>(); // Compiled into a constant '0', '1', etc.

    fill_layer(US_PAWN + idx, board.pieces(PT, US_COLOR));
    fill_layer(THEM_PAWN + idx, board.pieces(PT, THEM_COLOR));
  };

  fill_presence.template operator()<chess::PieceType::PAWN>();
  fill_presence.template operator()<chess::PieceType::KNIGHT>();
  fill_presence.template operator()<chess::PieceType::BISHOP>();
  fill_presence.template operator()<chess::PieceType::ROOK>();
  fill_presence.template operator()<chess::PieceType::QUEEN>();
  fill_presence.template operator()<chess::PieceType::KING>();

  // =============================================================
  //  MOVES & THREATS (12-27)
  // ============================================================

  // Board State & Caches
  auto us_pieces = board.us(US_COLOR);
  auto them_pieces = board.us(THEM_COLOR);
  auto occupied = board.occ();

  chess::Bitboard us_attacks_all = 0;
  chess::Bitboard them_attacks_all = 0;
  chess::Bitboard move_cache[12] = {0}; // [0-5]=Us Moves, [6-11]=Them Moves

  // --- PHASE 2: Legal Moves & Threats (12-25) ---
  chess::Bitboard ep_mask = 0;
  if (board.enpassantSq() != chess::Square::NO_SQ) {
    ep_mask = (1ULL << board.enpassantSq().index());
  }

  // Helper to generate moves for one side relative to perspective
  auto generate_moves = [&](chess::Color side, bool is_us) {
    int base_layer = is_us ? US_MOVE_PAWN : THEM_MOVE_PAWN;
    int cache_offset = is_us ? 0 : 6;
    chess::Bitboard &attack_acc = is_us ? us_attacks_all : them_attacks_all;
    chess::Bitboard enemies = is_us ? them_pieces : us_pieces;
    // only us can perform en passant
    chess::Bitboard ep_mask_effective = is_us ? ep_mask : 0;

    // A. Pawns
    {
      chess::Bitboard pawns = board.pieces(chess::PieceType::PAWN, side);
      chess::Bitboard push1, push2, attacks;

      if (side == chess::Color::WHITE) {
        push1 = (pawns << 8) & ~occupied;
        push2 = (push1 & 0x0000000000FF0000ULL) << 8 & ~occupied; // Rank 3 mask
        attacks = ((pawns & ~chess::Bitboard(chess::File::FILE_A)) << 7) |
                  ((pawns & ~chess::Bitboard(chess::File::FILE_H)) << 9);
      } else {
        push1 = (pawns >> 8) & ~occupied;
        push2 = (push1 & 0x0000FF0000000000ULL) >> 8 & ~occupied; // Rank 6 mask
        attacks = ((pawns & ~chess::Bitboard(chess::File::FILE_H)) >> 7) |
                  ((pawns & ~chess::Bitboard(chess::File::FILE_A)) >> 9);
      }

      attack_acc |= attacks;
      chess::Bitboard valid_moves =
          push1 | push2 | (attacks & (enemies | ep_mask_effective));
      move_cache[cache_offset + 0] = valid_moves;
      fill_layer(base_layer + 0, valid_moves);
    }

    auto process_piece = [&]<chess::PieceType::underlying PT>() {
      constexpr int i = get_index<PT>();

      // IMPORTANT: board.pieces() likely expects the 'PieceType' class wrapper.
      // We construct it from the enum using: chess::PieceType(PT)
      chess::Bitboard bb = board.pieces(chess::PieceType(PT), side);
      chess::Bitboard moves = 0;

      while (bb) {
        int sq = bb.pop();

        // Comparisons work directly because PT is the enum value
        if constexpr (PT == chess::PieceType::KNIGHT)
          moves |= chess::attacks::knight(sq);
        else if constexpr (PT == chess::PieceType::BISHOP)
          moves |= chess::attacks::bishop(sq, occupied);
        else if constexpr (PT == chess::PieceType::ROOK)
          moves |= chess::attacks::rook(sq, occupied);
        else if constexpr (PT == chess::PieceType::QUEEN)
          moves |= chess::attacks::queen(sq, occupied);
        else if constexpr (PT == chess::PieceType::KING)
          moves |= chess::attacks::king(sq);
      }

      attack_acc |= moves;
      move_cache[cache_offset + i] = moves;
      fill_layer(base_layer + i, moves);
    };

    // 3. Call it directly (No casting needed!)
    //    This works because chess::PieceType::KNIGHT is already of type
    //    'underlying'
    process_piece.template operator()<chess::PieceType::KNIGHT>();
    process_piece.template operator()<chess::PieceType::BISHOP>();
    process_piece.template operator()<chess::PieceType::ROOK>();
    process_piece.template operator()<chess::PieceType::QUEEN>();
    process_piece.template operator()<chess::PieceType::KING>();
  };

  generate_moves(US_COLOR, true);    // Us
  generate_moves(THEM_COLOR, false); // Them

  // Threats
  fill_layer(US_THREATS, us_attacks_all & (them_pieces | ep_mask));
  fill_layer(THEM_THREATS,
             them_attacks_all &
                 us_pieces); // Note: ep capture only possible by us

  // =============================================================
  //  CHECKS (26-27)
  // =============================================================
  chess::Square us_king = board.kingSq(US_COLOR);
  chess::Square them_king = board.kingSq(THEM_COLOR);

  auto get_checks = [&](chess::Color attacker,
                        chess::Square target_king) -> chess::Bitboard {
    if (target_king == chess::Square::NO_SQ)
      return 0;

    bool is_us_attacking = (attacker == US_COLOR);
    int cache_base = is_us_attacking ? 0 : 6;
    chess::Bitboard attacker_body = is_us_attacking ? us_pieces : them_pieces;

    // "Reverse" vision from King
    chess::Bitboard k_queen = chess::attacks::queen(target_king, occupied);
    chess::Bitboard k_rook = chess::attacks::rook(target_king, 0);
    chess::Bitboard diag = k_queen & ~k_rook;
    chess::Bitboard orth = k_queen & k_rook;
    chess::Bitboard k_knight = chess::attacks::knight(target_king);

    // Find pawns that attack king (by checking where a pawn on king sq would
    // attack)
    chess::Bitboard k_pawn = chess::attacks::pawn(
        (attacker == chess::Color::WHITE ? chess::Color::BLACK
                                         : chess::Color::WHITE),
        target_king);

    chess::Bitboard checks = 0;
    checks |= move_cache[cache_base + 0] & k_pawn;        // Pawn
    checks |= move_cache[cache_base + 1] & k_knight;      // Knight
    checks |= move_cache[cache_base + 2] & diag;          // Bishop
    checks |= move_cache[cache_base + 3] & orth;          // Rook
    checks |= move_cache[cache_base + 4] & (diag | orth); // Queen

    return checks & ~attacker_body;
  };

  fill_layer(US_CHECKS, get_checks(US_COLOR, them_king));
  fill_layer(THEM_CHECKS, get_checks(THEM_COLOR, us_king));

  // =============================================================
  //  DEFENDED PIECES (28-29)
  // =============================================================
  fill_layer(US_DEFENDED, us_attacks_all & us_pieces);
  fill_layer(THEM_DEFENDED, them_attacks_all & them_pieces);

  // =============================================================
  //  CONTINUOUS FEATURES: Distance from Kings (30-31)
  // =============================================================

  // Dense features use the pre-built Chebyshev tables which already account for
  // perspective. No need for orient_xor here - the tables are built correctly.

  if (us_king != chess::Square::NO_SQ) {
    const auto &cheb_table = CHEBYSHEV.data[CHEB_IDX][us_king.index()];
    std::memcpy(out.layers[US_KING_DIST], cheb_table, 64);
  }
  if (them_king != chess::Square::NO_SQ) {
    const auto &cheb_table = CHEBYSHEV.data[CHEB_IDX][them_king.index()];
    std::memcpy(out.layers[THEM_KING_DIST], cheb_table, 64);
  }

  // =============================================================
  //  SCALAR FEATURES
  // =============================================================

  // --- Castling Rights

  auto cr = board.castlingRights();
  using Side = chess::Board::CastlingRights::Side;
  out.global[US_OO] = cr.has(US_COLOR, Side::KING_SIDE);
  out.global[US_OOO] = cr.has(US_COLOR, Side::QUEEN_SIDE);
  out.global[THEM_OO] = cr.has(THEM_COLOR, Side::KING_SIDE);
  out.global[THEM_OOO] = cr.has(THEM_COLOR, Side::QUEEN_SIDE);

  // --- Material ---
  auto count = [&](chess::PieceType pt, chess::Color c) {
    return static_cast<float>(board.pieces(pt, c).count());
  };

  out.global[US_MAT_PAWN] = count(chess::PieceType::PAWN, US_COLOR) / 8.0f;
  out.global[THEM_MAT_PAWN] = count(chess::PieceType::PAWN, THEM_COLOR) / 8.0f;

  out.global[US_MAT_KNIGHT] = count(chess::PieceType::KNIGHT, US_COLOR) / 2.0f;
  out.global[THEM_MAT_KNIGHT] =
      count(chess::PieceType::KNIGHT, THEM_COLOR) / 2.0f;

  out.global[US_MAT_BISHOP] = count(chess::PieceType::BISHOP, US_COLOR) / 2.0f;
  out.global[THEM_MAT_BISHOP] =
      count(chess::PieceType::BISHOP, THEM_COLOR) / 2.0f;

  out.global[US_MAT_ROOK] = count(chess::PieceType::ROOK, US_COLOR) / 2.0f;
  out.global[THEM_MAT_ROOK] = count(chess::PieceType::ROOK, THEM_COLOR) / 2.0f;

  out.global[US_MAT_QUEEN] = count(chess::PieceType::QUEEN, US_COLOR);
  out.global[THEM_MAT_QUEEN] = count(chess::PieceType::QUEEN, THEM_COLOR);

  // =============================================================
  //  GAME PHASE (0.0 = Opening, 1.0 = Deep Endgame)
  // =============================================================
  // Uses "Mobility & Congestion" formula: P = 1 - (MaterialVitality * Omega) /
  // MaxPotential Optimized: reuse already-computed normalized material counts
  {
    // Weighted Material Vitality (Q=4, R=2, Minor=1, Pawn=0.2)
    float mat_score = 0.0f;
    mat_score += (out.global[US_MAT_QUEEN] + out.global[THEM_MAT_QUEEN]) *
                 1.0f * 4.0f; // Already raw counts
    mat_score += (out.global[US_MAT_ROOK] + out.global[THEM_MAT_ROOK]) * 2.0f *
                 2.0f; // De-normalize
    mat_score +=
        (out.global[US_MAT_BISHOP] + out.global[THEM_MAT_BISHOP]) * 2.0f * 1.0f;
    mat_score +=
        (out.global[US_MAT_KNIGHT] + out.global[THEM_MAT_KNIGHT]) * 2.0f * 1.0f;
    mat_score +=
        (out.global[US_MAT_PAWN] + out.global[THEM_MAT_PAWN]) * 8.0f * 0.2f;

    constexpr float MAX_MATERIAL =
        27.2f; // 2*Q(8) + 4*R(8) + 4*B(4) + 4*N(4) + 16*P(3.2)

    // Mobility from cached moves (already computed in move_cache)
    int total_moves = 0;
    for (int i = 0; i < 12; ++i) {
      total_moves += move_cache[i].count(); // __builtin_popcountll
    }

    constexpr float AVG_MIDGAME_MOVES = 75.0f;
    float omega = std::clamp(
        static_cast<float>(total_moves) / AVG_MIDGAME_MOVES, 0.5f, 1.5f);

    // Final phase: High material + high mobility = low phase (opening)
    float phase =
        1.0f - std::clamp((mat_score * omega) / MAX_MATERIAL, 0.0f, 1.0f);
    out.global[PHASE] = phase;
  }
}

void FeatureExtractor::fill_input(const chess::Board &board, ChessInput &out) {
  if (board.sideToMove() == chess::Color::WHITE) {
    fill_input_impl<chess::Color::WHITE>(board, out);
  } else {
    fill_input_impl<chess::Color::BLACK>(board, out);
  }
}