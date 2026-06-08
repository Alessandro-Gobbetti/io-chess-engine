/**
 * @file FactorizedFeatureExtractor.cpp
 * @brief Implementation of factorized feature extraction.
 */
#include "FactorizedFeatureExtractor.hpp"
#include <algorithm>
#include <array>
#include <cstring>

// ============================================================================
// Constants & Lookup Tables
// ============================================================================

// Pre-computed Chebyshev (king-distance) table, output as float ∈ [0, 1].
// Table[perspective][king_sq][output_sq]
//   perspective 0 = Black (no flip), 1 = White (rank-flip)
struct alignas(64) ChebTableF {
  float data[2][64][64];
};

static const auto CHEBYSHEV_F = []() {
  ChebTableF t{};
  for (int sq1 = 0; sq1 < 64; ++sq1) {
    int r1 = sq1 / 8, c1 = sq1 % 8;

    // Table 0: Black perspective (no flip)
    for (int sq2 = 0; sq2 < 64; ++sq2) {
      int r2 = sq2 / 8, c2 = sq2 % 8;
      int dist = std::max(std::abs(r1 - r2), std::abs(c1 - c2));
      t.data[0][sq1][sq2] = static_cast<float>(dist) / 7.0f;
    }

    // Table 1: White perspective (rank-flipped output positions)
    for (int out_sq = 0; out_sq < 64; ++out_sq) {
      int out_r = out_sq / 8, out_c = out_sq % 8;
      int sq2 = (7 - out_r) * 8 + out_c;
      int r2 = sq2 / 8, c2 = sq2 % 8;
      int dist = std::max(std::abs(r1 - r2), std::abs(c1 - c2));
      t.data[1][sq1][out_sq] = static_cast<float>(dist) / 7.0f;
    }
  }
  return t;
}();

// FRONT_SPAN[side][sq]: all squares in front of sq on same/adjacent files.
// side 0 = White forward (toward rank 8), side 1 = Black forward (toward rank 1).
struct FrontSpanTable {
  chess::Bitboard data[2][64];
};

static const auto FRONT_SPAN = []() {
  FrontSpanTable t{};
  for (int sq = 0; sq < 64; ++sq) {
    const int r = sq / 8;
    const int f = sq % 8;

    chess::Bitboard white_span = 0;
    for (int rr = r + 1; rr < 8; ++rr) {
      white_span.set(rr * 8 + f);
      if (f > 0)
        white_span.set(rr * 8 + f - 1);
      if (f < 7)
        white_span.set(rr * 8 + f + 1);
    }

    chess::Bitboard black_span = 0;
    for (int rr = r - 1; rr >= 0; --rr) {
      black_span.set(rr * 8 + f);
      if (f > 0)
        black_span.set(rr * 8 + f - 1);
      if (f < 7)
        black_span.set(rr * 8 + f + 1);
    }

    t.data[0][sq] = white_span;
    t.data[1][sq] = black_span;
  }
  return t;
}();

// BETWEEN_EXCLUSIVE[sq1][sq2]: all squares strictly between sq1 and sq2
// when aligned on rank/file/diagonal, otherwise empty.
static const auto BETWEEN_EXCLUSIVE = []() {
  std::array<std::array<chess::Bitboard, 64>, 64> t{};
  for (int sq1 = 0; sq1 < 64; ++sq1) {
    const int r1 = sq1 / 8;
    const int f1 = sq1 % 8;
    for (int sq2 = 0; sq2 < 64; ++sq2) {
      const int r2 = sq2 / 8;
      const int f2 = sq2 % 8;
      const int dr = r2 - r1;
      const int df = f2 - f1;

      const bool aligned = (dr == 0) || (df == 0) || (std::abs(dr) == std::abs(df));
      if (!aligned || (sq1 == sq2)) {
        continue;
      }

      const int step_r = (dr > 0) - (dr < 0);
      const int step_f = (df > 0) - (df < 0);
      int rr = r1 + step_r;
      int ff = f1 + step_f;
      while (rr != r2 || ff != f2) {
        t[(size_t)sq1][(size_t)sq2].set(rr * 8 + ff);
        rr += step_r;
        ff += step_f;
      }
    }
  }
  return t;
}();

// Full geometric slider rays from each square (ignore occupancy), excluding the
// source square itself. Used for rich X-ray planes.
struct SliderRayTable {
  chess::Bitboard data[64];
};

static const auto ROOK_XRAY_RAYS = []() {
  SliderRayTable t{};
  for (int sq = 0; sq < 64; ++sq) {
    const int r = sq / 8;
    const int f = sq % 8;

    chess::Bitboard rays = 0;
    for (int rr = r + 1; rr < 8; ++rr)
      rays.set(rr * 8 + f);
    for (int rr = r - 1; rr >= 0; --rr)
      rays.set(rr * 8 + f);
    for (int ff = f + 1; ff < 8; ++ff)
      rays.set(r * 8 + ff);
    for (int ff = f - 1; ff >= 0; --ff)
      rays.set(r * 8 + ff);

    t.data[sq] = rays;
  }
  return t;
}();

static const auto BISHOP_XRAY_RAYS = []() {
  SliderRayTable t{};
  for (int sq = 0; sq < 64; ++sq) {
    const int r = sq / 8;
    const int f = sq % 8;

    chess::Bitboard rays = 0;

    for (int rr = r + 1, ff = f + 1; rr < 8 && ff < 8; ++rr, ++ff)
      rays.set(rr * 8 + ff);
    for (int rr = r + 1, ff = f - 1; rr < 8 && ff >= 0; ++rr, --ff)
      rays.set(rr * 8 + ff);
    for (int rr = r - 1, ff = f + 1; rr >= 0 && ff < 8; --rr, ++ff)
      rays.set(rr * 8 + ff);
    for (int rr = r - 1, ff = f - 1; rr >= 0 && ff >= 0; --rr, --ff)
      rays.set(rr * 8 + ff);

    t.data[sq] = rays;
  }
  return t;
}();

static const auto QUEEN_XRAY_RAYS = []() {
  SliderRayTable t{};
  for (int sq = 0; sq < 64; ++sq) {
    t.data[sq] = ROOK_XRAY_RAYS.data[sq] | BISHOP_XRAY_RAYS.data[sq];
  }
  return t;
}();

// ============================================================================
// Piece-type → branch-offset mapping (compile-time)
// ============================================================================

template <chess::PieceType::underlying PT> constexpr int piece_offset() {
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
    return 0;
}

// ============================================================================
// Core extraction (side-to-move relative coordinates)
// ============================================================================

static void fill_factorized_impl(const chess::Board &board,
                                 FactorizedInput &out) {
  using FE = FactorizedFeatureExtractor;

  // ---- Side-to-move-relative setup ----
  const chess::Color us_color = board.sideToMove();
  const chess::Color them_color = ~us_color;
  const int orient_xor = (us_color == chess::Color::WHITE) ? 56 : 0;
  const int cheb_idx = (us_color == chess::Color::WHITE) ? 1 : 0;

  // ---- Zero everything ----
  std::memset(&out, 0, sizeof(out));

  // ---- Board state caches ----
  auto us_pieces = board.us(us_color);
  auto them_pieces = board.us(them_color);
  auto occupied = board.occ();
  int total_moves = 0;

  // ---- Helpers ----
  // Write 1.0f to a specific plane within a branch group
  auto set_branch = [&](int group, int plane, int sq) {
    out.branches[group][plane][sq ^ orient_xor] = 1.0f;
  };

  // Fill an entire bitboard into a branch plane
  auto fill_branch = [&](int group, int plane, chess::Bitboard bb) {
    while (bb) {
      set_branch(group, plane, bb.pop());
    }
  };

  // ================================================================
  //  PIECE PRESENCE (plane 0 of each branch)
  // ================================================================

  auto fill_presence = [&]<chess::PieceType::underlying PT>() {
    constexpr int off = piece_offset<PT>();
    fill_branch(FE::US_PAWN + off, FE::PRESENCE,
                board.pieces(PT, us_color));
    fill_branch(FE::THEM_PAWN + off, FE::PRESENCE,
                board.pieces(PT, them_color));
  };

  fill_presence.template operator()<chess::PieceType::PAWN>();
  fill_presence.template operator()<chess::PieceType::KNIGHT>();
  fill_presence.template operator()<chess::PieceType::BISHOP>();
  fill_presence.template operator()<chess::PieceType::ROOK>();
  fill_presence.template operator()<chess::PieceType::QUEEN>();
  fill_presence.template operator()<chess::PieceType::KING>();

  // ================================================================
  //  MOBILITY / ATTACKS / DEFENDS (planes 1-3 of each branch)
  // ================================================================
  //
  //  For each piece group on each side we compute:
  //    mobility = all pseudo-legal destination squares
  //    attacks  = mobility & enemy_pieces
  //    defends  = mobility & friendly_pieces  (x-ray defense)
  //
  // ================================================================

  // En-passant mask (only side-to-move can capture en passant)
  chess::Bitboard ep_mask = 0;
  if (board.enpassantSq() != chess::Square::NO_SQ) {
    ep_mask = (1ULL << board.enpassantSq().index());
  }

  // Process one side's pieces and fill branches [base..base+5]
  auto generate_side = [&](chess::Color side, bool is_us) {
    int base = is_us ? FE::US_PAWN : FE::THEM_PAWN;
    chess::Bitboard friendly = is_us ? us_pieces : them_pieces;
    chess::Bitboard enemies = is_us ? them_pieces : us_pieces;
    chess::Bitboard ep_effective =
        (board.sideToMove() == side) ? ep_mask : chess::Bitboard(0);

    // --- A. Pawns ---
    {
      chess::Bitboard pawns = board.pieces(chess::PieceType::PAWN, side);
      chess::Bitboard push1, push2, attacks;

      if (side == chess::Color::WHITE) {
        push1 = (pawns << 8) & ~occupied;
        push2 =
            (push1 & chess::Bitboard(0x0000000000FF0000ULL)) << 8 & ~occupied;
        attacks = ((pawns & ~chess::Bitboard(chess::File::FILE_A)) << 7) |
                  ((pawns & ~chess::Bitboard(chess::File::FILE_H)) << 9);
      } else {
        push1 = (pawns >> 8) & ~occupied;
        push2 =
            (push1 & chess::Bitboard(0x0000FF0000000000ULL)) >> 8 & ~occupied;
        attacks = ((pawns & ~chess::Bitboard(chess::File::FILE_H)) >> 7) |
                  ((pawns & ~chess::Bitboard(chess::File::FILE_A)) >> 9);
      }

      chess::Bitboard valid =
          push1 | push2 | (attacks & (enemies | ep_effective));
      chess::Bitboard captures = attacks & enemies;
      chess::Bitboard defends_bb = attacks & friendly;
      total_moves += valid.count();

      int g = base + 0; // pawn group
      fill_branch(g, FE::MOBILITY, valid);
      fill_branch(g, FE::ATTACKS, captures);
      fill_branch(g, FE::DEFENDS, defends_bb);
    }

    // --- B. Sliding & leaping pieces ---
    auto process_piece = [&]<chess::PieceType::underlying PT>() {
      constexpr int off = piece_offset<PT>();
      int g = base + off;

      chess::Bitboard bb = board.pieces(chess::PieceType(PT), side);
      chess::Bitboard moves_all = 0;

      while (bb) {
        int sq = bb.pop();
        chess::Bitboard moves = 0;

        if constexpr (PT == chess::PieceType::KNIGHT)
          moves = chess::attacks::knight(sq);
        else if constexpr (PT == chess::PieceType::BISHOP)
          moves = chess::attacks::bishop(sq, occupied);
        else if constexpr (PT == chess::PieceType::ROOK)
          moves = chess::attacks::rook(sq, occupied);
        else if constexpr (PT == chess::PieceType::QUEEN)
          moves = chess::attacks::queen(sq, occupied);
        else if constexpr (PT == chess::PieceType::KING)
          moves = chess::attacks::king(sq);

        moves_all |= moves;
      }
      total_moves += moves_all.count();

      fill_branch(g, FE::MOBILITY, moves_all);
      fill_branch(g, FE::ATTACKS, moves_all & enemies);
      fill_branch(g, FE::DEFENDS, moves_all & friendly);
    };

    process_piece.template operator()<chess::PieceType::KNIGHT>();
    process_piece.template operator()<chess::PieceType::BISHOP>();
    process_piece.template operator()<chess::PieceType::ROOK>();
    process_piece.template operator()<chess::PieceType::QUEEN>();
    process_piece.template operator()<chess::PieceType::KING>();
  };

  generate_side(us_color, true);
  generate_side(them_color, false);

  // ================================================================
  //  BYPASS PLANES — King Chebyshev Distance (continuous [0,1])
  // ================================================================

  chess::Square us_king = board.kingSq(us_color);
  chess::Square them_king = board.kingSq(them_color);

  if (us_king != chess::Square::NO_SQ) {
    std::memcpy(out.bypass[FE::US_KING_DIST],
                CHEBYSHEV_F.data[cheb_idx][us_king.index()],
                64 * sizeof(float));
  }
  if (them_king != chess::Square::NO_SQ) {
    std::memcpy(out.bypass[FE::THEM_KING_DIST],
                CHEBYSHEV_F.data[cheb_idx][them_king.index()],
                64 * sizeof(float));
  }

  // ================================================================
  //  GLOBAL SCALARS (identical logic to FeatureExtractor)
  // ================================================================

  // --- Castling Rights ---
  auto cr = board.castlingRights();
  using Side = chess::Board::CastlingRights::Side;
  out.global[FE::US_OO] = cr.has(us_color, Side::KING_SIDE);
  out.global[FE::US_OOO] = cr.has(us_color, Side::QUEEN_SIDE);
  out.global[FE::THEM_OO] = cr.has(them_color, Side::KING_SIDE);
  out.global[FE::THEM_OOO] = cr.has(them_color, Side::QUEEN_SIDE);

  // --- Material (normalized) ---
  auto count = [&](chess::PieceType pt, chess::Color c) {
    return static_cast<float>(board.pieces(pt, c).count());
  };

  out.global[FE::US_MAT_PAWN] = count(chess::PieceType::PAWN, us_color) / 8.0f;
  out.global[FE::THEM_MAT_PAWN] =
      count(chess::PieceType::PAWN, them_color) / 8.0f;

  out.global[FE::US_MAT_KNIGHT] =
    count(chess::PieceType::KNIGHT, us_color) / 2.0f;
  out.global[FE::THEM_MAT_KNIGHT] =
    count(chess::PieceType::KNIGHT, them_color) / 2.0f;

  out.global[FE::US_MAT_BISHOP] =
    count(chess::PieceType::BISHOP, us_color) / 2.0f;
  out.global[FE::THEM_MAT_BISHOP] =
    count(chess::PieceType::BISHOP, them_color) / 2.0f;

  out.global[FE::US_MAT_ROOK] = count(chess::PieceType::ROOK, us_color) / 2.0f;
  out.global[FE::THEM_MAT_ROOK] =
      count(chess::PieceType::ROOK, them_color) / 2.0f;

  out.global[FE::US_MAT_QUEEN] = count(chess::PieceType::QUEEN, us_color);
  out.global[FE::THEM_MAT_QUEEN] = count(chess::PieceType::QUEEN, them_color);

  // --- Game Phase ---
  {
    float mat_score = 0.0f;
    mat_score +=
        (out.global[FE::US_MAT_QUEEN] + out.global[FE::THEM_MAT_QUEEN]) *
        1.0f * 4.0f;
    mat_score +=
        (out.global[FE::US_MAT_ROOK] + out.global[FE::THEM_MAT_ROOK]) * 2.0f *
        2.0f;
    mat_score +=
        (out.global[FE::US_MAT_BISHOP] + out.global[FE::THEM_MAT_BISHOP]) *
        2.0f * 1.0f;
    mat_score +=
        (out.global[FE::US_MAT_KNIGHT] + out.global[FE::THEM_MAT_KNIGHT]) *
        2.0f * 1.0f;
    mat_score +=
        (out.global[FE::US_MAT_PAWN] + out.global[FE::THEM_MAT_PAWN]) * 8.0f *
        0.2f;

    constexpr float MAX_MATERIAL = 27.2f;

    constexpr float AVG_MIDGAME_MOVES = 75.0f;
    float omega = std::clamp(static_cast<float>(total_moves) / AVG_MIDGAME_MOVES,
                             0.5f, 1.5f);
    float phase =
        1.0f - std::clamp((mat_score * omega) / MAX_MATERIAL, 0.0f, 1.0f);
    out.global[FE::PHASE] = phase;
  }

  out.global[FE::IS_WHITE_TO_MOVE] =
      (board.sideToMove() == chess::Color::WHITE) ? 1.0f : 0.0f;
}

// ============================================================================
// Core extraction (Rich) — Variable planes per branch + 12 bypass + 32 globals
// ============================================================================
//
// Plane counts per piece type:
//   Pawn 4, Knight 4, Bishop 5, Rook 5, Queen 5, King 4
//

static void fill_factorized_rich_impl(const chess::Board &board,
                                      FactorizedInput &out) {
  using FE = FactorizedFeatureExtractor;

  // ---- Side-to-move-relative setup ----
  const chess::Color us_color = board.sideToMove();
  const chess::Color them_color = ~us_color;
  const int orient_xor = (us_color == chess::Color::WHITE) ? 56 : 0;
  const int cheb_idx = (us_color == chess::Color::WHITE) ? 1 : 0;

  // ---- Zero everything ----
  std::memset(&out, 0, sizeof(out));

  // ---- Board state caches ----
  auto us_pieces = board.us(us_color);
  auto them_pieces = board.us(them_color);
  auto occupied = board.occ();
  auto us_pawns = board.pieces(chess::PieceType::PAWN, us_color);
  auto them_pawns = board.pieces(chess::PieceType::PAWN, them_color);
  int total_moves = 0;

  // ---- Global heatmaps / masks accumulation ----
  chess::Bitboard pinned_us = 0;
  chess::Bitboard pinned_them = 0;
  chess::Bitboard safe_mobility_map = 0;
  chess::Bitboard king_ring_map = 0;
  chess::Bitboard passed_pawn_map = 0;
  chess::Bitboard outpost_map = 0;
  chess::Bitboard open_lines_map = 0;

  // 1. Pawn attacks for bypasses and mobility safety.
  auto pawn_attacks = [&](chess::Color side, chess::Bitboard pawns) {
    if (!pawns) {
      return chess::Bitboard(0);
    }
    if (side == chess::Color::WHITE) {
      return ((pawns & ~chess::Bitboard(chess::File::FILE_A)) << 7) |
             ((pawns & ~chess::Bitboard(chess::File::FILE_H)) << 9);
    }
    return ((pawns & ~chess::Bitboard(chess::File::FILE_H)) >> 7) |
           ((pawns & ~chess::Bitboard(chess::File::FILE_A)) >> 9);
  };

  chess::Bitboard us_pawn_attacks = pawn_attacks(us_color, us_pawns);
  chess::Bitboard them_pawn_attacks = pawn_attacks(them_color, them_pawns);

  chess::Bitboard ep_mask = 0;
  if (board.enpassantSq() != chess::Square::NO_SQ) {
    ep_mask = (1ULL << board.enpassantSq().index());
  }

  // 2. King rings.
  chess::Bitboard us_king_ring = 0;
  chess::Bitboard them_king_ring = 0;
  if (board.kingSq(us_color) != chess::Square::NO_SQ) {
    us_king_ring = chess::attacks::king(board.kingSq(us_color));
  }
  if (board.kingSq(them_color) != chess::Square::NO_SQ) {
    them_king_ring = chess::attacks::king(board.kingSq(them_color));
  }

  // 3. Absolute pins.
  auto calculate_pins = [&](chess::Color side, chess::Bitboard &pinned_bb) {
    chess::Square ksq = board.kingSq(side);
    if (ksq == chess::Square::NO_SQ) return;
    chess::Color enemy_side = ~side;
    chess::Bitboard attackers = (board.pieces(chess::PieceType::BISHOP, enemy_side) |
                                 board.pieces(chess::PieceType::ROOK, enemy_side) |
                                 board.pieces(chess::PieceType::QUEEN, enemy_side));
    while (attackers) {
      int asq = attackers.pop();
      chess::Bitboard b = BETWEEN_EXCLUSIVE[(size_t)asq][(size_t)ksq.index()];
      if (b && (b & occupied).count() == 1) {
        pinned_bb |= (b & occupied & board.us(side));
      }
    }
  };
  calculate_pins(us_color, pinned_us);
  calculate_pins(them_color, pinned_them);

  // 4. Passed pawn map (board-wide bypass map).
  auto accumulate_passed = [&](chess::Color side, chess::Bitboard pawns,
                               chess::Bitboard enemy_pawns) {
    const int side_idx = (side == chess::Color::WHITE) ? 0 : 1;
    chess::Bitboard pcopy = pawns;
    while (pcopy) {
      const int sq = pcopy.pop();
      if (FRONT_SPAN.data[side_idx][sq] & enemy_pawns)
        continue;
      passed_pawn_map.set(sq);

      // Mark immediate block square to expose passer race geometry.
      const int r = sq / 8;
      const int f = sq % 8;
      if (side == chess::Color::WHITE && r < 7)
        passed_pawn_map.set((r + 1) * 8 + f);
      else if (side == chess::Color::BLACK && r > 0)
        passed_pawn_map.set((r - 1) * 8 + f);
    }
  };
  accumulate_passed(us_color, us_pawns, them_pawns);
  accumulate_passed(them_color, them_pawns, us_pawns);

  // 5. Outpost map: squares defended by own pawns and not attacked by enemy pawns.
  outpost_map = (us_pawn_attacks & ~them_pawn_attacks) |
                (them_pawn_attacks & ~us_pawn_attacks);

  // 6. Open lines map: files with no pawns or pawns of only one color.
  for (int f = 0; f < 8; ++f) {
    const chess::Bitboard file_mask = chess::attacks::MASK_FILE[f];
    const bool us_on_file = static_cast<bool>(us_pawns & file_mask);
    const bool them_on_file = static_cast<bool>(them_pawns & file_mask);
    if (!(us_on_file && them_on_file))
      open_lines_map |= file_mask;
  }

  // ---- Helpers ----
  auto set_branch = [&](int group, int plane, int sq) {
    out.branches[group][plane][sq ^ orient_xor] = 1.0f;
  };

  auto fill_branch = [&](int group, int plane, chess::Bitboard bb) {
    while (bb) {
      set_branch(group, plane, bb.pop());
    }
  };

  // ================================================================
  //  PIECE PROCESSING (lean branch planes)
  // ================================================================

  auto generate_side = [&](chess::Color side, bool is_us) {
    int base = is_us ? FE::US_PAWN : FE::THEM_PAWN;
    chess::Bitboard friendly = is_us ? us_pieces : them_pieces;
    chess::Bitboard enemies = is_us ? them_pieces : us_pieces;
    chess::Bitboard opponent_pawn_attacks = is_us ? them_pawn_attacks : us_pawn_attacks;

    // --- A. Pawns (4 planes) ---
    {
      int g = base + 0;
      chess::Bitboard pawns = is_us ? us_pawns : them_pawns;
      fill_branch(g, FE::PRESENCE, pawns);

      chess::Bitboard push1, push2, attacks;
      if (side == chess::Color::WHITE) {
        push1 = (pawns << 8) & ~occupied;
        push2 = (push1 & chess::Bitboard(0x0000000000FF0000ULL)) << 8 & ~occupied;
        attacks = ((pawns & ~chess::Bitboard(chess::File::FILE_A)) << 7) |
                  ((pawns & ~chess::Bitboard(chess::File::FILE_H)) << 9);
      } else {
        push1 = (pawns >> 8) & ~occupied;
        push2 = (push1 & chess::Bitboard(0x0000FF0000000000ULL)) >> 8 & ~occupied;
        attacks = ((pawns & ~chess::Bitboard(chess::File::FILE_H)) >> 7) |
                  ((pawns & ~chess::Bitboard(chess::File::FILE_A)) >> 9);
      }

      chess::Bitboard mobility = push1 | push2 | (attacks & enemies);
      fill_branch(g, FE::MOBILITY, mobility);
      fill_branch(g, FE::ATTACKS, attacks & enemies);
      fill_branch(g, FE::DEFENDS, attacks & friendly);
      safe_mobility_map |= (mobility & ~opponent_pawn_attacks);
      king_ring_map |= (attacks & (is_us ? them_king_ring : us_king_ring));

      chess::Bitboard ep_eff = (side == board.sideToMove()) ? ep_mask : chess::Bitboard(0);
      total_moves += (push1 | push2 | (attacks & (enemies | ep_eff))).count();
    }

    // --- B. Knights (4 planes: Presence, Mobility, Attacks, Defends) ---
    {
      int g = base + 1;
      chess::Bitboard knights = board.pieces(chess::PieceType::KNIGHT, side);
      fill_branch(g, FE::PRESENCE, knights);

      chess::Bitboard mobility_all = 0;
      chess::Bitboard attacks_all = 0;
      chess::Bitboard defends_all = 0;

      while (knights) {
        int sq = knights.pop();
        chess::Bitboard moves = chess::attacks::knight(sq);
        mobility_all |= moves;
        attacks_all |= (moves & enemies);
        defends_all |= (moves & friendly);
        total_moves += moves.count();
      }

      fill_branch(g, FE::MOBILITY, mobility_all);
      fill_branch(g, FE::ATTACKS, attacks_all);
      fill_branch(g, FE::DEFENDS, defends_all);
      safe_mobility_map |= (mobility_all & ~opponent_pawn_attacks);
      king_ring_map |= (attacks_all & (is_us ? them_king_ring : us_king_ring));
    }

    // --- C. Bishops (5 planes: 0-4, includes X-ray) ---
    {
      int g = base + 2;
      chess::Bitboard bishops = board.pieces(chess::PieceType::BISHOP, side);
      fill_branch(g, FE::PRESENCE, bishops);

      chess::Bitboard mobility_all = 0;
      chess::Bitboard attacks_all = 0;
      chess::Bitboard defends_all = 0;
      chess::Bitboard xray_all = 0;

      while (bishops) {
        int sq = bishops.pop();
        chess::Bitboard moves = chess::attacks::bishop(sq, occupied);
        mobility_all |= moves;
        attacks_all |= (moves & enemies);
        defends_all |= (moves & friendly);
        total_moves += moves.count();

        // Full geometric X-ray rays (precomputed), unioned across bishops.
        xray_all |= BISHOP_XRAY_RAYS.data[sq];
      }

      fill_branch(g, FE::MOBILITY, mobility_all);
      fill_branch(g, FE::ATTACKS, attacks_all);
      fill_branch(g, FE::DEFENDS, defends_all);
      fill_branch(g, FE::X_RAY, xray_all);
      safe_mobility_map |= (mobility_all & ~opponent_pawn_attacks);
      king_ring_map |= (attacks_all & (is_us ? them_king_ring : us_king_ring));
    }

    // --- D. Rooks (5 planes: 0-4, includes X-ray) ---
    {
      int g = base + 3;
      chess::Bitboard rooks = board.pieces(chess::PieceType::ROOK, side);
      fill_branch(g, FE::PRESENCE, rooks);

      chess::Bitboard mobility_all = 0;
      chess::Bitboard attacks_all = 0;
      chess::Bitboard defends_all = 0;
      chess::Bitboard xray_all = 0;

      while (rooks) {
        int sq = rooks.pop();
        chess::Bitboard moves = chess::attacks::rook(sq, occupied);
        mobility_all |= moves;
        attacks_all |= (moves & enemies);
        defends_all |= (moves & friendly);
        total_moves += moves.count();

        // Full geometric X-ray rays (precomputed), unioned across rooks.
        xray_all |= ROOK_XRAY_RAYS.data[sq];
      }

      fill_branch(g, FE::MOBILITY, mobility_all);
      fill_branch(g, FE::ATTACKS, attacks_all);
      fill_branch(g, FE::DEFENDS, defends_all);
      fill_branch(g, FE::X_RAY, xray_all);
      safe_mobility_map |= (mobility_all & ~opponent_pawn_attacks);
      king_ring_map |= (attacks_all & (is_us ? them_king_ring : us_king_ring));
    }

    // --- E. Queens (5 planes: 0-4 shared, unchanged) ---
    {
      int g = base + 4;
      chess::Bitboard queens = board.pieces(chess::PieceType::QUEEN, side);
      fill_branch(g, FE::PRESENCE, queens);

      chess::Bitboard mobility_all = 0;
      chess::Bitboard attacks_all = 0;
      chess::Bitboard defends_all = 0;
      chess::Bitboard xray_all = 0;

      while (queens) {
        int sq = queens.pop();
        chess::Bitboard moves = chess::attacks::queen(sq, occupied);
        mobility_all |= moves;
        attacks_all |= (moves & enemies);
        defends_all |= (moves & friendly);
        total_moves += moves.count();

        // Full geometric X-ray rays (precomputed), unioned across queens.
        xray_all |= QUEEN_XRAY_RAYS.data[sq];
      }

      fill_branch(g, FE::MOBILITY, mobility_all);
      fill_branch(g, FE::ATTACKS, attacks_all);
      fill_branch(g, FE::DEFENDS, defends_all);
      fill_branch(g, FE::X_RAY, xray_all);
      safe_mobility_map |= (mobility_all & ~opponent_pawn_attacks);
      king_ring_map |= (attacks_all & (is_us ? them_king_ring : us_king_ring));
    }

    // --- F. King (4 planes: 0-3 shared) ---
    {
      int g = base + 5;
      chess::Square ksq = board.kingSq(side);
      if (ksq == chess::Square::NO_SQ) return;

      fill_branch(g, FE::PRESENCE, chess::Bitboard(1ULL << ksq.index()));

      chess::Bitboard moves = chess::attacks::king(ksq);
      fill_branch(g, FE::MOBILITY, moves);
      fill_branch(g, FE::ATTACKS, moves & enemies);
      fill_branch(g, FE::DEFENDS, moves & friendly);
      safe_mobility_map |= (moves & ~opponent_pawn_attacks);
      king_ring_map |= ((moves & enemies) & (is_us ? them_king_ring : us_king_ring));
      total_moves += moves.count();
    }
  };

  generate_side(us_color, true);
  generate_side(them_color, false);

  // ================================================================
  //  BYPASS PLANES
  // ================================================================

  // 1. King Dist
  chess::Square us_king = board.kingSq(us_color);
  chess::Square them_king = board.kingSq(them_color);
  if (us_king != chess::Square::NO_SQ)
    std::memcpy(out.bypass[FE::US_KING_DIST], CHEBYSHEV_F.data[cheb_idx][us_king.index()], 64 * sizeof(float));
  if (them_king != chess::Square::NO_SQ)
    std::memcpy(out.bypass[FE::THEM_KING_DIST], CHEBYSHEV_F.data[cheb_idx][them_king.index()], 64 * sizeof(float));

  // 2. Pawn Attacks
  auto fill_bypass_bb = [&](int idx, chess::Bitboard bb) {
    while (bb) {
      out.bypass[idx][bb.pop() ^ orient_xor] = 1.0f;
    }
  };
  fill_bypass_bb(FE::US_PAWN_ATTACKS, us_pawn_attacks);
  fill_bypass_bb(FE::THEM_PAWN_ATTACKS, them_pawn_attacks);

  // 3. Legacy TOTAL_CONTROL slot now mirrors open-lines map to keep layout
  // stable while removing expensive per-square control accumulation.
  fill_bypass_bb(FE::TOTAL_CONTROL, open_lines_map);

  // 4. Absolute Pins
  fill_bypass_bb(FE::ABSOLUTE_PINS, pinned_us | pinned_them);

  // 5. En Prise / Hanging
  chess::Bitboard all_us = us_pieces;
  chess::Bitboard all_them = them_pieces;
  chess::Bitboard kings_mask = board.pieces(chess::PieceType::KING);

  chess::Bitboard us_pawn_attacked = all_us & them_pawn_attacks;
  chess::Bitboard them_pawn_attacked = all_them & us_pawn_attacks;
  chess::Bitboard us_pawn_defended = all_us & us_pawn_attacks;
  chess::Bitboard them_pawn_defended = all_them & them_pawn_attacks;

  chess::Bitboard en_prise = (us_pawn_attacked & ~us_pawn_defended) |
                             (them_pawn_attacked & ~them_pawn_defended);
  en_prise &= ~kings_mask;
  fill_bypass_bb(FE::EN_PRISE, en_prise);

  // 6. Safe Mobility Map
  fill_bypass_bb(FE::SAFE_MOBILITY_MAP, safe_mobility_map);

  // 7. King Ring Map
  fill_bypass_bb(FE::KING_RING_MAP, king_ring_map);

  // 8. Passed Pawn Map
  fill_bypass_bb(FE::PASSED_PAWN_MAP, passed_pawn_map);

  // 9. Board-wide outpost squares
  fill_bypass_bb(FE::OUTPOST_MAP, outpost_map);

  // 10. Open/semi-open lines map
  fill_bypass_bb(FE::OPEN_LINES_MAP, open_lines_map);

  // ================================================================
  //  GLOBAL SCALARS
  // ================================================================
  auto cr = board.castlingRights();
  using Side = chess::Board::CastlingRights::Side;
  out.global[FE::US_OO] = cr.has(us_color, Side::KING_SIDE);
  out.global[FE::US_OOO] = cr.has(us_color, Side::QUEEN_SIDE);
  out.global[FE::THEM_OO] = cr.has(them_color, Side::KING_SIDE);
  out.global[FE::THEM_OOO] = cr.has(them_color, Side::QUEEN_SIDE);

  auto count = [&](chess::PieceType pt, chess::Color c) {
    return static_cast<float>(board.pieces(pt, c).count());
  };
  out.global[FE::US_MAT_PAWN] = count(chess::PieceType::PAWN, us_color) / 8.0f;
  out.global[FE::THEM_MAT_PAWN] = count(chess::PieceType::PAWN, them_color) / 8.0f;
  out.global[FE::US_MAT_KNIGHT] = count(chess::PieceType::KNIGHT, us_color) / 2.0f;
  out.global[FE::THEM_MAT_KNIGHT] = count(chess::PieceType::KNIGHT, them_color) / 2.0f;
  out.global[FE::US_MAT_BISHOP] = count(chess::PieceType::BISHOP, us_color) / 2.0f;
  out.global[FE::THEM_MAT_BISHOP] = count(chess::PieceType::BISHOP, them_color) / 2.0f;
  out.global[FE::US_MAT_ROOK] = count(chess::PieceType::ROOK, us_color) / 2.0f;
  out.global[FE::THEM_MAT_ROOK] = count(chess::PieceType::ROOK, them_color) / 2.0f;
  out.global[FE::US_MAT_QUEEN] = count(chess::PieceType::QUEEN, us_color);
  out.global[FE::THEM_MAT_QUEEN] = count(chess::PieceType::QUEEN, them_color);

  // Bishop Pair
  out.global[FE::US_BISHOP_PAIR] = board.pieces(chess::PieceType::BISHOP, us_color).count() >= 2;
  out.global[FE::THEM_BISHOP_PAIR] = board.pieces(chess::PieceType::BISHOP, them_color).count() >= 2;
  
  // Opposite Bishops
  if (board.pieces(chess::PieceType::BISHOP, us_color).count() == 1 && 
      board.pieces(chess::PieceType::BISHOP, them_color).count() == 1) {
    chess::Square b1 = chess::Square(board.pieces(chess::PieceType::BISHOP, us_color).lsb());
    chess::Square b2 = chess::Square(board.pieces(chess::PieceType::BISHOP, them_color).lsb());
    out.global[FE::OPPOSITE_BISHOPS] = (b1.is_light() != b2.is_light());
  }

  // Rook vs 2 Minors
  auto is_rook_vs_2m = [&](chess::Color side) {
    return (board.pieces(chess::PieceType::ROOK, side).count() >= 1 && 
            (board.pieces(chess::PieceType::KNIGHT, ~side).count() + board.pieces(chess::PieceType::BISHOP, ~side).count()) >= 2);
  };
  out.global[FE::US_ROOK_VS_MINORS] = is_rook_vs_2m(us_color);
  out.global[FE::THEM_ROOK_VS_MINORS] = is_rook_vs_2m(them_color);

    // Phase (material weighted by current mobility density).
    float mat_score = 0.0f;
    mat_score +=
      (out.global[FE::US_MAT_PAWN] + out.global[FE::THEM_MAT_PAWN]) * 8.0f * 1.0f;
    mat_score +=
      (out.global[FE::US_MAT_KNIGHT] + out.global[FE::THEM_MAT_KNIGHT]) * 2.0f * 3.0f;
    mat_score +=
      (out.global[FE::US_MAT_BISHOP] + out.global[FE::THEM_MAT_BISHOP]) * 2.0f * 3.1f;
    mat_score +=
      (out.global[FE::US_MAT_ROOK] + out.global[FE::THEM_MAT_ROOK]) * 2.0f * 5.0f;
    mat_score +=
      (out.global[FE::US_MAT_QUEEN] + out.global[FE::THEM_MAT_QUEEN]) * 9.0f;

    constexpr float MAX_MATERIAL = 27.2f;
    constexpr float AVG_MIDGAME_MOVES = 75.0f;
    float omega = std::clamp(static_cast<float>(total_moves) / AVG_MIDGAME_MOVES,
                 0.5f, 1.5f);
    out.global[FE::PHASE] =
      1.0f - std::clamp((mat_score * omega) / MAX_MATERIAL, 0.0f, 1.0f);
  out.global[FE::IS_WHITE_TO_MOVE] =
      (board.sideToMove() == chess::Color::WHITE) ? 1.0f : 0.0f;
}

void FactorizedFeatureExtractor::fill_input_rich(const chess::Board &board,
                                                 FactorizedInput &out) {
  fill_factorized_rich_impl(board, out);
}

void FactorizedFeatureExtractor::fill_input(const chess::Board &board,
                                            FactorizedInput &out) {
  fill_factorized_impl(board, out);
}
