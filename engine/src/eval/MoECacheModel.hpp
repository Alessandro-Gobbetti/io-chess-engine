/**
 * @file MoECacheModel.hpp
 * @brief Factorized Mixture of Experts (MoE) neural network architecture and
 * inference components.
 * @ingroup engine
 *
 * This file implements the core inference structures for the natively-compiled
 * Factorized MoE network used for position evaluation. The architecture
 * processes a custom 12-branch spatial feature set extracted via
 * `FactorizedFeatureExtractor`, applies a mixer layer, routes the evaluation to
 * specialized expert networks, and computes the final Win/Draw/Loss
 * probabilities.
 *
 * ## Architecture Overview
 * - **Branches**: 12 independent 3x3 convolutional feature branches (e.g.,
 * White Pawns, Black Knights, etc.).
 * - **Mixer**: A fully-connected layer that combines the outputs of all 12
 * branches and global features.
 * - **Router**: A shallow network that selects the most appropriate "Expert"
 * sub-networks for the given position (e.g., Endgame vs Opening).
 * - **Experts**: Specialized fully-connected networks (Bottleneck -> Hidden ->
 * WDL Output).
 *
 * ## Double Accumulator (Incremental Inference)
 * For maximum performance, this file uses a double accumulator pattern
 * (`MoEDoubleAccumulator`):
 * 1. **Base Accumulator**: Stores the state of the network for the current
 * position.
 * 2. **Dirty Accumulator**: When a move is made, instead of running the entire
 * network, we only compute the delta (difference) for the specific pieces that
 * moved or were captured (`dirty_branches`). This reduces the evaluation time
 * from ~20 microseconds (full forward pass) to <1 microsecond (incremental
 * update).
 *
 * @note This file contains both engine evaluation structures
 * (`MoEDoubleAccumulator`) and full-batch multi-threaded evaluation structures
 * (`SharedMoEWeights::forward`) which are strictly retained for the standalone
 * benchmark tools in `src/tools/`.
 */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if defined(__SSE3__)
#include <pmmintrin.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <chess.hpp>

#include "FactorizedFeatureExtractor.hpp"

using namespace chess;
using Clock = std::chrono::high_resolution_clock;

static constexpr int kDefaultBranchDim = 16;
static constexpr int kDefaultMixerOut = 64;
static constexpr int kDefaultBypass = 12;
static constexpr int kDefaultGlobals = 21;
static constexpr int kDefaultExperts = 4;
static constexpr int kDefaultExpertBottleneck = 32;
static constexpr int kDefaultExpertHidden = 128;

static constexpr int NET_BRANCH_DIM = 16;
static constexpr int NET_MIXER_OUT = 64;
static constexpr int NET_BYPASS = 12;
static constexpr int NET_GLOBALS = 21;
static constexpr int NET_EXPERTS = 4;
static constexpr int NET_EXPERT_BOTTLENECK = 32;
static constexpr int NET_EXPERT_HIDDEN = 128;

// FactorizedInput currently exposes 12 bypass planes and 32 global scalars.
static constexpr int kInputBypassPlanes = 12;
static constexpr int kInputGlobals = 32;

// Upper bounds for runtime-configurable dimensions while keeping static aligned
// storage in the hot path.
static constexpr int kMaxBranchDim = NET_BRANCH_DIM;
static constexpr int kMaxMixerOut = NET_MIXER_OUT;
static constexpr int kMaxBypass = NET_BYPASS;
static constexpr int kMaxGlobals = NET_GLOBALS;
static constexpr int kMaxExperts = NET_EXPERTS;
static constexpr int kMaxExpertBottleneck = NET_EXPERT_BOTTLENECK;
static constexpr int kMaxExpertHidden = NET_EXPERT_HIDDEN;
static constexpr int kMixerOcTile = 8;

#if defined(__clang__)
#define FORCE_VECTORIZE                                                        \
  _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__) || defined(__GNUG__)
#define FORCE_VECTORIZE _Pragma("GCC ivdep")
#else
#define FORCE_VECTORIZE
#endif

#if defined(_MSC_VER)
#define HOT_RESTRICT __restrict
#else
#define HOT_RESTRICT __restrict__
#endif

static inline double us(Clock::time_point a, Clock::time_point b) {
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
             .count() /
         1000.0;
}

#include "PersistentThreadPool.hpp"

/**
 * @brief Determines if a specific branch corresponds to a slider piece type
 * (Bishop, Rook, Queen).
 * @param branch_idx The index of the branch (0-11).
 * @return True if the branch represents a slider piece, false otherwise.
 */
static inline bool is_slider_branch_rich(int branch_idx) {
  switch (branch_idx) {
  case 2:
  case 3:
  case 4:
  case 8:
  case 9:
  case 10:
    return true;
  default:
    return false;
  }
}

/**
 * @brief Returns the number of active input feature channels for a given
 * branch.
 * @param branch_idx The index of the branch.
 * @return 5 channels for sliders (includes X-ray features), 4 channels for
 * non-sliders.
 */
static inline int active_channels_for_branch(int branch_idx) {
  // Rich feature policy:
  // - Sliders (B/R/Q): 5 planes (includes X-ray)
  // - Knights: 4 planes
  // - Pawns/Kings: 4 planes
  if (is_slider_branch_rich(branch_idx))
    return 5;
  return 4;
}

/**
 * @brief Checks if a slider piece type (Bishop, Rook, Queen) has become
 * "dirty".
 *
 * A slider is dirty if any piece of that type moved, or if the board occupancy
 * changed in a way that intersects with any of the slider's rays (blocking or
 * unblocking its attack path).
 *
 * @param board The new board state.
 * @param old_board The previous board state.
 * @param side The color of the slider pieces to check.
 * @param pt The piece type (BISHOP, ROOK, or QUEEN).
 * @return True if the slider's features need to be recomputed, false otherwise.
 */
static bool is_slider_dirty(const Board &board, const Board &old_board,
                            Color side, PieceType pt) {
  Bitboard current_sliders = board.pieces(pt, side);
  Bitboard old_sliders = old_board.pieces(pt, side);
  if (current_sliders != old_sliders)
    return true;

  Bitboard changed_occupancy = board.occ() ^ old_board.occ();
  if (!changed_occupancy)
    return false;

  Bitboard sliders_copy = current_sliders;
  while (sliders_copy) {
    Square sq = Square(sliders_copy.pop());
    Bitboard rays = 0;
    if (pt == PieceType::BISHOP || pt == PieceType::QUEEN)
      rays |= attacks::bishop(sq, board.occ());
    if (pt == PieceType::ROOK || pt == PieceType::QUEEN)
      rays |= attacks::rook(sq, board.occ());
    if (rays & changed_occupancy)
      return true;
  }
  return false;
}

/**
 * @brief Marks a specific piece type for a specific color as "dirty" in the
 * dirty mask.
 * @param m The 12-element boolean array mapping to the 12 spatial branches.
 * @param s The color of the piece.
 * @param pt The type of the piece.
 */
static void mark_head(std::array<uint8_t, 12> &m, Color s, PieceType pt) {
  int b = (s == Color::WHITE) ? 0 : 6;
  int i = -1;
  if (pt == PieceType::PAWN)
    i = b;
  else if (pt == PieceType::KNIGHT)
    i = b + 1;
  else if (pt == PieceType::BISHOP)
    i = b + 2;
  else if (pt == PieceType::ROOK)
    i = b + 3;
  else if (pt == PieceType::QUEEN)
    i = b + 4;
  else if (pt == PieceType::KING)
    i = b + 5;
  if (i >= 0)
    m[(size_t)i] = 1;
}

/**
 * @brief Analyzes a chess move to determine exactly which of the 12 spatial
 * branches need recomputing.
 *
 * This is the core of the incremental update logic. It prevents full network
 * re-evaluations by figuring out which specific piece types (e.g., White Pawns,
 * Black Knights) had their board representation changed by the move, including
 * complex cases like discoveries, castling, and en passant.
 *
 * @param old_board The board state before the move.
 * @param new_board The board state after the move.
 * @param mv The move that was just played.
 * @return An array of 12 bytes acting as boolean flags (1 = dirty, 0 = clean)
 * for each branch.
 */
static std::array<uint8_t, 12> build_dirty_mask(const Board &old_board,
                                                const Board &new_board,
                                                const Move &mv) {
  std::array<uint8_t, 12> m{};
  Color mover = old_board.sideToMove();
  Color enemy = ~mover;

  Piece p = old_board.at(mv.from());
  if (p == Piece::NONE) {
    m.fill(1);
    return m;
  }

  mark_head(m, mover, p.type());
  if (mv.typeOf() == Move::PROMOTION)
    mark_head(m, mover, mv.promotionType());
  if (mv.typeOf() == Move::CASTLING)
    mark_head(m, mover, PieceType::ROOK);

  Square cs = Square::NO_SQ;
  Piece cap = Piece::NONE;
  if (mv.typeOf() == Move::ENPASSANT) {
    cs = mv.to().ep_square();
    cap = Piece(PieceType::PAWN, enemy);
  } else if (old_board.isCapture(mv)) {
    cs = mv.to();
    cap = old_board.at(mv.to());
  }
  if (cap != Piece::NONE)
    mark_head(m, enemy, cap.type());

  auto king_touch = [](Square sq, Square k) {
    if (sq == Square::NO_SQ || k == Square::NO_SQ)
      return true;
    return static_cast<bool>((attacks::king(k) | Bitboard(1ULL << k.index())) &
                             Bitboard(1ULL << sq.index()));
  };

  Square wk = old_board.kingSq(Color::WHITE);
  Square bk = old_board.kingSq(Color::BLACK);
  if (king_touch(mv.from(), wk) || king_touch(mv.to(), wk) ||
      king_touch(cs, wk) || p.type() == PieceType::KING) {
    mark_head(m, Color::WHITE, PieceType::KING);
  }
  if (king_touch(mv.from(), bk) || king_touch(mv.to(), bk) ||
      king_touch(cs, bk) || p.type() == PieceType::KING) {
    mark_head(m, Color::BLACK, PieceType::KING);
  }

  // Tight slider invalidation: mark only if slider moved/captured or ray
  // changed.
  for (auto side : {Color::WHITE, Color::BLACK}) {
    if (is_slider_dirty(new_board, old_board, side, PieceType::BISHOP))
      mark_head(m, side, PieceType::BISHOP);
    if (is_slider_dirty(new_board, old_board, side, PieceType::ROOK))
      mark_head(m, side, PieceType::ROOK);
    if (is_slider_dirty(new_board, old_board, side, PieceType::QUEEN))
      mark_head(m, side, PieceType::QUEEN);
  }

  return m;
}

/**
 * @brief Performs a fast byte-level comparison to see if two spatial branch
 * inputs are identical.
 * @param a The old factorized input.
 * @param b The new factorized input.
 * @param branch_idx The index of the branch to check.
 * @return True if the inputs differ, false if they are identical.
 */
static inline bool branch_planes_changed(const FactorizedInput &a,
                                         const FactorizedInput &b,
                                         int branch_idx) {
  const int ch = active_channels_for_branch(branch_idx);
  const size_t bytes = (size_t)ch * 64 * sizeof(float);
  return std::memcmp(&a.branches[branch_idx][0][0],
                     &b.branches[branch_idx][0][0], bytes) != 0;
}

/**
 * @brief SIMD-optimized fused multiply-add (FMA) for adding a scaled vector to
 * a destination vector. dst[i] += src[i] * scale
 * @param dst The accumulator array.
 * @param src The source array to scale and add.
 * @param scale The scalar multiplier.
 * @param n Number of elements.
 */
static inline void simd_add_scaled(float *HOT_RESTRICT dst,
                                   const float *HOT_RESTRICT src, float scale,
                                   int n) {
  FORCE_VECTORIZE
  for (int i = 0; i < n; ++i)
    dst[i] += src[i] * scale;
}

/**
 * @brief Hints the compiler that a pointer is aligned to a 32-byte boundary
 * (for AVX operations).
 */
template <typename T> static inline T *assume_aligned_32(T *ptr) {
#if defined(__clang__) || defined(__GNUC__)
  return static_cast<T *>(__builtin_assume_aligned(ptr, 32));
#else
  return ptr;
#endif
}

template <typename T> static inline const T *assume_aligned_32(const T *ptr) {
#if defined(__clang__) || defined(__GNUC__)
  return static_cast<const T *>(__builtin_assume_aligned(ptr, 32));
#else
  return ptr;
#endif
}

/**
 * @brief Computes a full 3x3 2D convolution for a single input channel and
 * accumulates it into `out_plane`.
 *
 * This avoids boundary checks by breaking the 8x8 chessboard into regions
 * (Center, North/South edges, Corners).
 *
 * @param out_plane The destination 64-element array.
 * @param in_plane The source 64-element array representing the spatial
 * features.
 * @param wk A 9-element array representing the 3x3 convolution weights.
 */
static inline void conv3x3_accumulate_plane(float *HOT_RESTRICT out_plane,
                                            const float *HOT_RESTRICT in_plane,
                                            const float *HOT_RESTRICT wk) {
  // Center (64 contiguous cells)
  const float wC = wk[4];
  FORCE_VECTORIZE
  for (int sq = 0; sq < 64; ++sq)
    out_plane[sq] += in_plane[sq] * wC;

  // North/South as contiguous 1-D slices to avoid short-tail paths.
  const float wN = wk[1];
  FORCE_VECTORIZE
  for (int sq = 8; sq < 64; ++sq)
    out_plane[sq] += in_plane[sq - 8] * wN;

  const float wS = wk[7];
  FORCE_VECTORIZE
  for (int sq = 0; sq < 56; ++sq)
    out_plane[sq] += in_plane[sq + 8] * wS;

  const float wNW = wk[0];
  for (int r = 1; r < 8; ++r) {
    FORCE_VECTORIZE
    for (int c = 1; c < 8; ++c)
      out_plane[r * 8 + c] += in_plane[(r - 1) * 8 + (c - 1)] * wNW;
  }

  const float wNE = wk[2];
  for (int r = 1; r < 8; ++r) {
    FORCE_VECTORIZE
    for (int c = 0; c < 7; ++c)
      out_plane[r * 8 + c] += in_plane[(r - 1) * 8 + (c + 1)] * wNE;
  }

  const float wW = wk[3];
  for (int r = 0; r < 8; ++r) {
    FORCE_VECTORIZE
    for (int c = 1; c < 8; ++c)
      out_plane[r * 8 + c] += in_plane[r * 8 + (c - 1)] * wW;
  }

  const float wE = wk[5];
  for (int r = 0; r < 8; ++r) {
    FORCE_VECTORIZE
    for (int c = 0; c < 7; ++c)
      out_plane[r * 8 + c] += in_plane[r * 8 + (c + 1)] * wE;
  }

  const float wSW = wk[6];
  for (int r = 0; r < 7; ++r) {
    FORCE_VECTORIZE
    for (int c = 1; c < 8; ++c)
      out_plane[r * 8 + c] += in_plane[(r + 1) * 8 + (c - 1)] * wSW;
  }

  const float wSE = wk[8];
  for (int r = 0; r < 7; ++r) {
    FORCE_VECTORIZE
    for (int c = 0; c < 7; ++c)
      out_plane[r * 8 + c] += in_plane[(r + 1) * 8 + (c + 1)] * wSE;
  }
}

/**
 * @brief Computes a 3x3 convolution across multiple input channels (`ic`) to
 * produce a single output channel, followed by ReLU.
 * @param in The flattened input tensor [ic][64].
 * @param w The flattened weights [ic][9].
 * @param b The bias scalar.
 * @param out The destination 64-element array for the output channel.
 * @param ic Number of input channels.
 */
static inline void conv3x3_single_out_relu(const float *HOT_RESTRICT in,
                                           const float *HOT_RESTRICT w, float b,
                                           float *HOT_RESTRICT out, int ic) {
  FORCE_VECTORIZE
  for (int sq = 0; sq < 64; ++sq)
    out[sq] = b;

  for (int i = 0; i < ic; ++i) {
    const float *wk = &w[(size_t)i * 9];
    const float *in_plane = &in[(size_t)i * 64];
    conv3x3_accumulate_plane(out, in_plane, wk);
  }

  FORCE_VECTORIZE
  for (int sq = 0; sq < 64; ++sq)
    out[sq] = std::max(0.0f, out[sq]);
}

/**
 * @brief Computes a full 3x3 convolution from `ic` input channels to `oc`
 * output channels, followed by ReLU.
 * @param in Flattened input tensor [ic][64].
 * @param w Flattened weights [oc][ic][9].
 * @param b Flattened biases [oc].
 * @param out Flattened output tensor [oc][64].
 * @param ic Number of input channels.
 * @param oc Number of output channels.
 */
static void conv3x3_relu(const float *HOT_RESTRICT in,
                         const float *HOT_RESTRICT w,
                         const float *HOT_RESTRICT b, float *HOT_RESTRICT out,
                         int ic, int oc) {
  const float *aligned_in = assume_aligned_32(in);
  const float *aligned_w = assume_aligned_32(w);
  const float *aligned_b = assume_aligned_32(b);
  float *aligned_out = assume_aligned_32(out);

  for (int o = 0; o < oc; ++o) {
    float *out_plane = &aligned_out[(size_t)o * 64];
    conv3x3_single_out_relu(aligned_in, &aligned_w[(size_t)o * ic * 9],
                            aligned_b[o], out_plane, ic);
  }
}

static inline void conv3x3_relu_bd16_dispatch(const float *HOT_RESTRICT in,
                                              const float *HOT_RESTRICT w,
                                              const float *HOT_RESTRICT b,
                                              float *HOT_RESTRICT out, int ic) {
  conv3x3_relu(in, w, b, out, ic, 16);
}

#if defined(__ARM_NEON)
static inline float32x4_t neon_fma_n(float32x4_t acc, float32x4_t x, float w) {
#if defined(__aarch64__)
  return vfmaq_n_f32(acc, x, w);
#else
  return vmlaq_n_f32(acc, x, w);
#endif
}

static inline void conv3x3_row_accumulate_4oc_l0(
    const float *w0, const float *w1, const float *w2, const float *w3,
    const float *in_row, const float32x4_t &vzero, float32x4_t &a00,
    float32x4_t &a01, float32x4_t &a10, float32x4_t &a11, float32x4_t &a20,
    float32x4_t &a21, float32x4_t &a30, float32x4_t &a31, int kr) {
  const float32x4_t v_row_0 = vld1q_f32(in_row);
  const float32x4_t v_row_1 = vld1q_f32(in_row + 4);
  const float32x4_t v_l0 = vextq_f32(vzero, v_row_0, 3);
  const float32x4_t v_l1 = vextq_f32(v_row_0, v_row_1, 3);
  const float32x4_t v_r0 = vextq_f32(v_row_0, v_row_1, 1);
  const float32x4_t v_r1 = vextq_f32(v_row_1, vzero, 1);

  const int base = kr * 3;
  const float wl00 = w0[base + 0], wc00 = w0[base + 1], wr00 = w0[base + 2];
  const float wl10 = w1[base + 0], wc10 = w1[base + 1], wr10 = w1[base + 2];
  const float wl20 = w2[base + 0], wc20 = w2[base + 1], wr20 = w2[base + 2];
  const float wl30 = w3[base + 0], wc30 = w3[base + 1], wr30 = w3[base + 2];

  a00 = neon_fma_n(a00, v_l0, wl00);
  a01 = neon_fma_n(a01, v_l1, wl00);
  a00 = neon_fma_n(a00, v_row_0, wc00);
  a01 = neon_fma_n(a01, v_row_1, wc00);
  a00 = neon_fma_n(a00, v_r0, wr00);
  a01 = neon_fma_n(a01, v_r1, wr00);

  a10 = neon_fma_n(a10, v_l0, wl10);
  a11 = neon_fma_n(a11, v_l1, wl10);
  a10 = neon_fma_n(a10, v_row_0, wc10);
  a11 = neon_fma_n(a11, v_row_1, wc10);
  a10 = neon_fma_n(a10, v_r0, wr10);
  a11 = neon_fma_n(a11, v_r1, wr10);

  a20 = neon_fma_n(a20, v_l0, wl20);
  a21 = neon_fma_n(a21, v_l1, wl20);
  a20 = neon_fma_n(a20, v_row_0, wc20);
  a21 = neon_fma_n(a21, v_row_1, wc20);
  a20 = neon_fma_n(a20, v_r0, wr20);
  a21 = neon_fma_n(a21, v_r1, wr20);

  a30 = neon_fma_n(a30, v_l0, wl30);
  a31 = neon_fma_n(a31, v_l1, wl30);
  a30 = neon_fma_n(a30, v_row_0, wc30);
  a31 = neon_fma_n(a31, v_row_1, wc30);
  a30 = neon_fma_n(a30, v_r0, wr30);
  a31 = neon_fma_n(a31, v_r1, wr30);
}

static inline void conv3x3_relu_bd16_neon(const float *in, const float *w,
                                          const float *b, float *out,
                                          int ic_total) {
  const float32x4_t vzero = vdupq_n_f32(0.0f);

  for (int oc = 0; oc < 16; oc += 4) {
    for (int r = 0; r < 8; ++r) {
      float32x4_t a00 = vdupq_n_f32(b[oc + 0]);
      float32x4_t a01 = vdupq_n_f32(b[oc + 0]);
      float32x4_t a10 = vdupq_n_f32(b[oc + 1]);
      float32x4_t a11 = vdupq_n_f32(b[oc + 1]);
      float32x4_t a20 = vdupq_n_f32(b[oc + 2]);
      float32x4_t a21 = vdupq_n_f32(b[oc + 2]);
      float32x4_t a30 = vdupq_n_f32(b[oc + 3]);
      float32x4_t a31 = vdupq_n_f32(b[oc + 3]);

      for (int ic = 0; ic < ic_total; ++ic) {
        const float *in_plane = &in[(size_t)ic * 64];
        const float *w0 = &w[((size_t)(oc + 0) * ic_total + (size_t)ic) * 9];
        const float *w1 = &w[((size_t)(oc + 1) * ic_total + (size_t)ic) * 9];
        const float *w2 = &w[((size_t)(oc + 2) * ic_total + (size_t)ic) * 9];
        const float *w3 = &w[((size_t)(oc + 3) * ic_total + (size_t)ic) * 9];

        if (r == 0) {
          conv3x3_row_accumulate_4oc_l0(w0, w1, w2, w3, &in_plane[0], vzero,
                                        a00, a01, a10, a11, a20, a21, a30, a31,
                                        1);
          conv3x3_row_accumulate_4oc_l0(w0, w1, w2, w3, &in_plane[8], vzero,
                                        a00, a01, a10, a11, a20, a21, a30, a31,
                                        2);
        } else if (r == 7) {
          conv3x3_row_accumulate_4oc_l0(w0, w1, w2, w3, &in_plane[48], vzero,
                                        a00, a01, a10, a11, a20, a21, a30, a31,
                                        0);
          conv3x3_row_accumulate_4oc_l0(w0, w1, w2, w3, &in_plane[56], vzero,
                                        a00, a01, a10, a11, a20, a21, a30, a31,
                                        1);
        } else {
          conv3x3_row_accumulate_4oc_l0(
              w0, w1, w2, w3, &in_plane[(size_t)(r - 1) * 8], vzero, a00, a01,
              a10, a11, a20, a21, a30, a31, 0);
          conv3x3_row_accumulate_4oc_l0(w0, w1, w2, w3,
                                        &in_plane[(size_t)r * 8], vzero, a00,
                                        a01, a10, a11, a20, a21, a30, a31, 1);
          conv3x3_row_accumulate_4oc_l0(
              w0, w1, w2, w3, &in_plane[(size_t)(r + 1) * 8], vzero, a00, a01,
              a10, a11, a20, a21, a30, a31, 2);
        }
      }

      const size_t row_base = (size_t)r * 8;
      vst1q_f32(&out[(size_t)(oc + 0) * 64 + row_base + 0],
                vmaxq_f32(vzero, a00));
      vst1q_f32(&out[(size_t)(oc + 0) * 64 + row_base + 4],
                vmaxq_f32(vzero, a01));
      vst1q_f32(&out[(size_t)(oc + 1) * 64 + row_base + 0],
                vmaxq_f32(vzero, a10));
      vst1q_f32(&out[(size_t)(oc + 1) * 64 + row_base + 4],
                vmaxq_f32(vzero, a11));
      vst1q_f32(&out[(size_t)(oc + 2) * 64 + row_base + 0],
                vmaxq_f32(vzero, a20));
      vst1q_f32(&out[(size_t)(oc + 2) * 64 + row_base + 4],
                vmaxq_f32(vzero, a21));
      vst1q_f32(&out[(size_t)(oc + 3) * 64 + row_base + 0],
                vmaxq_f32(vzero, a30));
      vst1q_f32(&out[(size_t)(oc + 3) * 64 + row_base + 4],
                vmaxq_f32(vzero, a31));
    }
  }
}
#endif

/**
 * @brief Computes a depthwise 3x3 convolution followed by a ReLU activation.
 *
 * In a depthwise convolution, each input channel is convolved with its own set
 * of spatial weights, producing exactly one output channel per input channel,
 * without mixing information across channels.
 *
 * @param in Flattened input tensor [channels][64].
 * @param w Flattened depthwise weights [channels][9].
 * @param b Flattened biases [channels].
 * @param out Flattened output tensor [channels][64].
 * @param channels Number of channels (both input and output).
 */
static inline void depthwise_conv3x3_relu(const float *HOT_RESTRICT in,
                                          const float *HOT_RESTRICT w,
                                          const float *HOT_RESTRICT b,
                                          float *HOT_RESTRICT out,
                                          int channels) {
  for (int c = 0; c < channels; ++c) {
    const float *in_plane = &in[(size_t)c * 64];
    float *out_plane = &out[(size_t)c * 64];
    const float *wk = &w[(size_t)c * 9];

    FORCE_VECTORIZE
    for (int sq = 0; sq < 64; ++sq)
      out_plane[sq] = b[c];

    conv3x3_accumulate_plane(out_plane, in_plane, wk);

    FORCE_VECTORIZE
    for (int sq = 0; sq < 64; ++sq)
      out_plane[sq] = std::max(0.0f, out_plane[sq]);
  }
}

/**
 * @brief Computes a 1x1 convolution (pointwise convolution) across the spatial
 * grid, followed by ReLU.
 *
 * This operation mixes features across channels for every spatial square
 * independently.
 *
 * @param in Flattened input tensor [ic][64].
 * @param w Flattened weights [oc][ic].
 * @param b Flattened biases [oc].
 * @param out Flattened output tensor [oc][64].
 * @param ic Number of input channels.
 * @param oc Number of output channels.
 */
static void conv1x1_relu(const float *HOT_RESTRICT in,
                         const float *HOT_RESTRICT w,
                         const float *HOT_RESTRICT b, float *HOT_RESTRICT out,
                         int ic, int oc) {
  const float *aligned_in = assume_aligned_32(in);
  const float *aligned_w = assume_aligned_32(w);
  const float *aligned_b = assume_aligned_32(b);
  float *aligned_out = assume_aligned_32(out);

  // 1) Initialize output with bias.
  for (int o = 0; o < oc; ++o) {
    const float bias = aligned_b[o];
    float *out_plane = &aligned_out[(size_t)o * 64];
    FORCE_VECTORIZE
    for (int sq = 0; sq < 64; ++sq)
      out_plane[sq] = bias;
  }

  // 2) Accumulate 1x1 products.
  for (int o = 0; o < oc; ++o) {
    float *out_plane = &aligned_out[(size_t)o * 64];
    for (int i = 0; i < ic; ++i) {
      const float weight = aligned_w[(size_t)o * ic + i];
      const float *in_plane = &aligned_in[(size_t)i * 64];
      FORCE_VECTORIZE
      for (int sq = 0; sq < 64; ++sq)
        out_plane[sq] += in_plane[sq] * weight;
    }
  }

  // 3) Apply ReLU.
  for (int o = 0; o < oc; ++o) {
    float *out_plane = &aligned_out[(size_t)o * 64];
    FORCE_VECTORIZE
    for (int sq = 0; sq < 64; ++sq)
      out_plane[sq] = std::max(0.0f, out_plane[sq]);
  }
}

#if defined(__ARM_NEON)
static inline void conv1x1_bd16_relu_neon(const float *HOT_RESTRICT in,
                                          const float *HOT_RESTRICT w,
                                          const float *HOT_RESTRICT b,
                                          float *HOT_RESTRICT out) {
  const float32x4_t vzero = vdupq_n_f32(0.0f);

  for (int oc = 0; oc < 16; oc += 4) {
    const float *w0 = &w[(size_t)(oc + 0) * 16];
    const float *w1 = &w[(size_t)(oc + 1) * 16];
    const float *w2 = &w[(size_t)(oc + 2) * 16];
    const float *w3 = &w[(size_t)(oc + 3) * 16];

    for (int sq = 0; sq < 64; sq += 4) {
      float32x4_t acc0 = vdupq_n_f32(b[oc + 0]);
      float32x4_t acc1 = vdupq_n_f32(b[oc + 1]);
      float32x4_t acc2 = vdupq_n_f32(b[oc + 2]);
      float32x4_t acc3 = vdupq_n_f32(b[oc + 3]);

      for (int ic = 0; ic < 16; ++ic) {
        const float32x4_t vin = vld1q_f32(&in[(size_t)ic * 64 + (size_t)sq]);
#if defined(__aarch64__)
        acc0 = vfmaq_n_f32(acc0, vin, w0[ic]);
        acc1 = vfmaq_n_f32(acc1, vin, w1[ic]);
        acc2 = vfmaq_n_f32(acc2, vin, w2[ic]);
        acc3 = vfmaq_n_f32(acc3, vin, w3[ic]);
#else
        acc0 = vmlaq_n_f32(acc0, vin, w0[ic]);
        acc1 = vmlaq_n_f32(acc1, vin, w1[ic]);
        acc2 = vmlaq_n_f32(acc2, vin, w2[ic]);
        acc3 = vmlaq_n_f32(acc3, vin, w3[ic]);
#endif
      }

      vst1q_f32(&out[(size_t)(oc + 0) * 64 + (size_t)sq],
                vmaxq_f32(vzero, acc0));
      vst1q_f32(&out[(size_t)(oc + 1) * 64 + (size_t)sq],
                vmaxq_f32(vzero, acc1));
      vst1q_f32(&out[(size_t)(oc + 2) * 64 + (size_t)sq],
                vmaxq_f32(vzero, acc2));
      vst1q_f32(&out[(size_t)(oc + 3) * 64 + (size_t)sq],
                vmaxq_f32(vzero, acc3));
    }
  }
}
#endif

#if defined(__AVX2__)
static inline __m256 avx2_fmadd_ps(__m256 a, __m256 b, __m256 c) {
#if defined(__FMA__)
  return _mm256_fmadd_ps(a, b, c);
#else
  return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

static inline void conv1x1_bd16_relu_avx2(const float *HOT_RESTRICT in,
                                          const float *HOT_RESTRICT w,
                                          const float *HOT_RESTRICT b,
                                          float *HOT_RESTRICT out) {
  const __m256 vzero = _mm256_setzero_ps();
  assert((reinterpret_cast<uintptr_t>(in) & 31u) == 0u);
  assert((reinterpret_cast<uintptr_t>(out) & 31u) == 0u);

  for (int oc = 0; oc < 16; oc += 4) {
    const float *w0 = &w[(size_t)(oc + 0) * 16];
    const float *w1 = &w[(size_t)(oc + 1) * 16];
    const float *w2 = &w[(size_t)(oc + 2) * 16];
    const float *w3 = &w[(size_t)(oc + 3) * 16];

    for (int sq = 0; sq < 64; sq += 8) {
      __m256 acc0 = _mm256_set1_ps(b[oc + 0]);
      __m256 acc1 = _mm256_set1_ps(b[oc + 1]);
      __m256 acc2 = _mm256_set1_ps(b[oc + 2]);
      __m256 acc3 = _mm256_set1_ps(b[oc + 3]);

      for (int ic = 0; ic < 16; ++ic) {
        const __m256 vin = _mm256_load_ps(&in[(size_t)ic * 64 + (size_t)sq]);
        acc0 = avx2_fmadd_ps(vin, _mm256_set1_ps(w0[ic]), acc0);
        acc1 = avx2_fmadd_ps(vin, _mm256_set1_ps(w1[ic]), acc1);
        acc2 = avx2_fmadd_ps(vin, _mm256_set1_ps(w2[ic]), acc2);
        acc3 = avx2_fmadd_ps(vin, _mm256_set1_ps(w3[ic]), acc3);
      }

      _mm256_store_ps(&out[(size_t)(oc + 0) * 64 + (size_t)sq],
                      _mm256_max_ps(vzero, acc0));
      _mm256_store_ps(&out[(size_t)(oc + 1) * 64 + (size_t)sq],
                      _mm256_max_ps(vzero, acc1));
      _mm256_store_ps(&out[(size_t)(oc + 2) * 64 + (size_t)sq],
                      _mm256_max_ps(vzero, acc2));
      _mm256_store_ps(&out[(size_t)(oc + 3) * 64 + (size_t)sq],
                      _mm256_max_ps(vzero, acc3));
    }
  }
}
#endif

/**
 * @enum ExpertPoolMode
 * @brief Defines how spatial features are pooled before entering the
 * fully-connected expert networks.
 */
enum class ExpertPoolMode {
  Flat,
  Gap,
  Pool2x2Avg,
  Pool2x2Max,
};

static constexpr int kPool2x2Regions = 16;

static inline const char *expert_pool_mode_name(ExpertPoolMode mode) {
  switch (mode) {
  case ExpertPoolMode::Flat:
    return "flat";
  case ExpertPoolMode::Gap:
    return "gap";
  case ExpertPoolMode::Pool2x2Avg:
    return "pool2avg";
  case ExpertPoolMode::Pool2x2Max:
    return "pool2max";
  }
  return "flat";
}

static inline int pool2x2_region_from_sq(int sq) {
  const int r = sq >> 3;
  const int c = sq & 7;
  return ((r >> 1) << 2) | (c >> 1);
}

static inline void pool2x2_region_base(int region, int &r0, int &c0) {
  r0 = (region >> 2) * 2;
  c0 = (region & 3) * 2;
}

struct BenchConfig {
  int nGames = 20;
  int nPlies = 40;
  unsigned seed = 42;
  int nThreads = 1;
  int minParallelDirtyHeads = 4;
  int minParallelActiveExperts = 3;
  int denseDirtySqThreshold = 16;
  ExpertPoolMode expertPoolMode = ExpertPoolMode::Flat;
  bool routeSlowGlobals = false;

  int branchConvLayers = 3;
  int branchDim = kDefaultBranchDim;
  int mixerOut = kDefaultMixerOut;
  int nBypass = kDefaultBypass;
  int nGlobals = kDefaultGlobals;
  int nExperts = kDefaultExperts;
  int expertBottleneck = kDefaultExpertBottleneck;
  int expertHidden = kDefaultExpertHidden;
};

struct BenchResult {
  std::string name;
  double fullUs = 0.0;
  long long fullPlies = 0;
  double incUs = 0.0;
  long long incPlies = 0;
  long long incDirtyHeads = 0;
  long long incEvents = 0;

  // Pre-generation costs (outside timed inference loop).
  double prepPlayUs = 0.0;
  double prepFeatureUs = 0.0;
  double prepDirtyUs = 0.0;

  // Timed inference breakdown.
  double fullRouteUs = 0.0;
  double fullUpdateUs = 0.0;
  double fullExpertUs = 0.0;
  double incRouteUs = 0.0;
  double incUpdateUs = 0.0;
  double incExpertUs = 0.0;

  // Full rebuild internal breakdown.
  double fullBranchForwardUs = 0.0;
  double fullMixerAccumUs = 0.0;
  double fullMixerReluUs = 0.0;
  double fullExpertCacheUs = 0.0;

  // Incremental update internal breakdown.
  double incBranchDeltaUs = 0.0;
  double incBypassDeltaUs = 0.0;
  double incGlobalReluUs = 0.0;
  double incExpertBottleneckUs = 0.0;
  double incHiddenDeltaUs = 0.0;
  double incExpertCacheRebuildUs = 0.0;

  double fullNps() const {
    return fullUs > 0 ? (fullPlies * 1e6 / fullUs) : 0.0;
  }
  double incNps() const { return incUs > 0 ? (incPlies * 1e6 / incUs) : 0.0; }
  double avgDirtyHeads() const {
    return incEvents > 0 ? (double)incDirtyHeads / (double)incEvents : 0.0;
  }
};

struct BranchLayer {
  int ic = 0;
  int oc = 0;
  alignas(32) std::array<float, (size_t)5 * NET_BRANCH_DIM * 9> w{};
  alignas(32) std::array<float, NET_BRANCH_DIM> b{};
};

struct Branch {
  BranchLayer l0;
  BranchLayer l1;
  BranchLayer l2;
};

struct Expert {
  alignas(32) std::array<float, (size_t)NET_EXPERT_BOTTLENECK *
                                    NET_MIXER_OUT> wConv{};     // [ebo][nf]
  alignas(32) std::array<float, NET_EXPERT_BOTTLENECK> bConv{}; // [ebo]

  alignas(32)
      std::array<float, (size_t)NET_EXPERT_HIDDEN *
                            NET_EXPERT_BOTTLENECK> wHG{}; // [eh][ebo] (GAP
                                                          // pooled hidden path)
  alignas(32) std::array<
      float, (size_t)NET_EXPERT_BOTTLENECK *
                 NET_EXPERT_HIDDEN> wHGT{}; // [ebo][eh] (transposed wHG)

  alignas(
      32) std::array<float, (size_t)NET_EXPERT_HIDDEN * NET_EXPERT_BOTTLENECK *
                                kPool2x2Regions> wH16{}; // [eh][ebo*16] (2x2
                                                         // pooled hidden path)
  alignas(32) std::array<
      float, (size_t)NET_EXPERT_BOTTLENECK * kPool2x2Regions *
                 NET_EXPERT_HIDDEN> wH16T{}; // [ebo*16][eh] (transposed wH16)

  alignas(
      32) std::array<float, (size_t)NET_EXPERT_HIDDEN * NET_EXPERT_BOTTLENECK *
                                64> wH{}; // [eh][ebo*64]
  alignas(32) std::array<float, (size_t)NET_EXPERT_BOTTLENECK * 64 *
                                    NET_EXPERT_HIDDEN> wHT{}; // [ebo*64][eh]
                                                              // (transposed wH)
  alignas(32) std::array<float, NET_EXPERT_HIDDEN> bH{};      // [eh]

  alignas(
      32) std::array<float, (size_t)3 * NET_EXPERT_HIDDEN> wWdl{}; // [3][eh]
  alignas(32) std::array<float, 3> bWdl{};                         // [3]
};

/**
 * @struct SharedMoEWeights
 * @brief Contains the globally shared, read-only weights for the Factorized MoE
 * network.
 *
 * This struct stores the pre-trained weights for all convolution branches, the
 * mixer layer, the router gate, and the individual expert networks. In the
 * engine, these weights are loaded once into memory and shared across all
 * search threads.
 *
 * @note Includes `forward()`, a full-batch non-incremental evaluation function
 * strictly used by benchmark tools. The engine does not use `forward()`.
 */
struct SharedMoEWeights {
  static constexpr int bd = NET_BRANCH_DIM;
  static constexpr int nf = NET_MIXER_OUT;
  static constexpr int nBypass = NET_BYPASS;
  static constexpr int nGlobals = NET_GLOBALS;
  static constexpr int nExperts = NET_EXPERTS;
  static constexpr int ebo = NET_EXPERT_BOTTLENECK;
  static constexpr int eh = NET_EXPERT_HIDDEN;

  std::array<Branch, 12> branches{};

  alignas(64) std::array<float, (size_t)12 * NET_MIXER_OUT *
                                    NET_BRANCH_DIM> mixerWBr{}; // [12][nf][bd]
  alignas(64) std::array<float, (size_t)NET_BYPASS *
                                    NET_MIXER_OUT> mixerWBp{}; // [nBypass][nf]
  alignas(64) std::array<float, NET_MIXER_OUT> mixerB{};       // [nf]

  alignas(64) std::array<float, (size_t)NET_MIXER_OUT *
                                    NET_GLOBALS> globalW{}; // [nf][nGlobals]
  alignas(64) std::array<float, NET_MIXER_OUT> globalB{};   // [nf]

  alignas(64) std::array<
      float, (size_t)NET_EXPERTS * NET_GLOBALS> gateW{}; // [nExperts][nGlobals]
  alignas(64) std::array<float, NET_EXPERTS> gateB{};    // [nExperts]

  std::array<Expert, NET_EXPERTS> experts{};

  void init_architecture(int branchConvLayers) {
    auto init_branch = [&](int ic) {
      Branch br;
      br.l0.ic = ic;
      br.l0.oc = bd;

      if (branchConvLayers >= 2) {
        br.l1.ic = bd;
        br.l1.oc = bd;
      }

      if (branchConvLayers >= 3) {
        br.l2.ic = bd;
        br.l2.oc = bd;
      }
      return br;
    };

    for (int b = 0; b < 12; ++b)
      branches[(size_t)b] = init_branch(active_channels_for_branch(b));
  }
};

/**
 * @struct MoEDoubleAccumulator
 * @brief Thread-local state for incremental, lightning-fast neural network
 * inference.
 *
 * The double accumulator pattern avoids running the full neural network on
 * every position. Instead, it maintains a base state (`base_` variables)
 * corresponding to the parent position. When a move is played, it identifies
 * which feature branches changed (e.g. only a Knight moved) and calculates the
 * difference (`dirty_branches`). It then propagates only these differences
 * through the mixer and into the active expert networks.
 *
 * The `update_incremental()` function achieves <1 μs latency by leveraging this
 * delta propagation.
 */
struct MoEDoubleAccumulator {
  int branchConvLayers = 3;
  int nThreads = 1;
  int minParallelDirtyHeads = 4;
  int minParallelActiveExperts = 3;
  int denseDirtySqThreshold = 16;
  ExpertPoolMode expertPoolMode = ExpertPoolMode::Pool2x2Avg;
  bool routeSlowGlobals = false;
  static constexpr int bd = NET_BRANCH_DIM;
  static constexpr int nf = NET_MIXER_OUT;
  static constexpr int nBypass = NET_BYPASS;
  static constexpr int nGlobals = NET_GLOBALS;
  static constexpr int nExperts = NET_EXPERTS;
  static constexpr int ebo = NET_EXPERT_BOTTLENECK;
  static constexpr int eh = NET_EXPERT_HIDDEN;

  const SharedMoEWeights *weights = nullptr;
  std::shared_ptr<SharedMoEWeights> ownedWeights{};

  std::unique_ptr<PersistentThreadPool> threadPool;

  // Persistent caches for incremental path.
  alignas(64) std::array<float, (size_t)12 * kMaxBranchDim * 64> branchCache{};
  alignas(64) std::array<float, (size_t)kMaxMixerOut * 64> mixerLinearAccum{};
  alignas(64) std::array<float, (size_t)kMaxMixerOut * 64> mixerReluCache{};

  alignas(64) std::array<std::array<float, (size_t)kMaxExpertBottleneck * 64>,
                         kMaxExperts> exPreAccum{};
  alignas(64) std::array<std::array<float, (size_t)kMaxExpertBottleneck * 64>,
                         kMaxExperts> exReluCache{};
  std::array<uint8_t, kMaxExperts> exValid{};
  alignas(64)
      std::array<std::array<float, kMaxExpertHidden>, kMaxExperts> hiddenAcc{};
  alignas(64) std::array<std::array<float, kMaxExpertBottleneck>,
                         kMaxExperts> exGapCache{};
  alignas(64) std::array<
      std::array<float, (size_t)kMaxExpertBottleneck * kPool2x2Regions>,
      kMaxExperts> exPool16Cache{};
  alignas(64) std::array<float, kMaxGlobals> oldGlobalV{};

  // Scratch buffers (fixed-size, no heap allocations).
  alignas(64) std::array<float, (size_t)kMaxBranchDim * 64> scratchT0{};
  alignas(64) std::array<float, (size_t)kMaxBranchDim * 64> scratchT1{};
  alignas(64) std::array<float, (size_t)kMaxBranchDim * 64> scratchNewBranch{};
  alignas(64) std::array<std::array<float, (size_t)kMaxBranchDim * 64>,
                         12> scratchParallelBranch0{};
  alignas(64) std::array<std::array<float, (size_t)kMaxBranchDim * 64>,
                         12> scratchParallelBranch1{};
  alignas(64)
      std::array<float, (size_t)12 * kMaxBranchDim * 64> scratchDirtyBranches{};
  alignas(
      64) std::array<float, (size_t)kMaxBranchDim * 64> scratchBranchDelta{};
  alignas(64) std::array<float, (size_t)kMaxBypass * 64> scratchBypassDelta{};
  alignas(64) std::array<float, kMaxMixerOut> scratchGproj{};
  alignas(64) std::array<float, (size_t)kMaxMixerOut * 64> scratchDeltaRelu{};
  alignas(64)
      std::array<float, (size_t)kMaxExpertBottleneck * 64> scratchFlatDelta{};
  alignas(64) std::array<std::array<float, (size_t)kMaxExpertBottleneck * 64>,
                         kMaxExperts> scratchParallelExpertDelta{};
  alignas(64) std::array<float, kMaxExpertHidden> scratchHidden{};

  struct PhaseProfile {
    double fullBranchForwardUs = 0.0;
    double fullMixerAccumUs = 0.0;
    double fullMixerReluUs = 0.0;
    double fullExpertCacheUs = 0.0;

    double incBranchDeltaUs = 0.0;
    double incBypassDeltaUs = 0.0;
    double incGlobalReluUs = 0.0;
    double incExpertBottleneckUs = 0.0;
    double incHiddenDeltaUs = 0.0;
    double incExpertCacheRebuildUs = 0.0;
  };

  PhaseProfile profile{};

  bool initialized = false;

  const SharedMoEWeights &shared_weights() const {
    if (!weights)
      throw std::runtime_error("MoE weights pointer is null");
    return *weights;
  }

  SharedMoEWeights &mutable_owned_weights() {
    if (!ownedWeights)
      ownedWeights = std::make_shared<SharedMoEWeights>();
    weights = ownedWeights.get();
    return *ownedWeights;
  }

  void reset_runtime_state() {
    std::fill(branchCache.begin(), branchCache.end(), 0.0f);
    std::fill(mixerLinearAccum.begin(), mixerLinearAccum.end(), 0.0f);
    std::fill(mixerReluCache.begin(), mixerReluCache.end(), 0.0f);
    for (auto &v : exPreAccum)
      std::fill(v.begin(), v.end(), 0.0f);
    for (auto &v : exReluCache)
      std::fill(v.begin(), v.end(), 0.0f);
    std::fill(exValid.begin(), exValid.end(), 0);
    for (auto &v : hiddenAcc)
      std::fill(v.begin(), v.end(), 0.0f);
    for (auto &v : exGapCache)
      std::fill(v.begin(), v.end(), 0.0f);
    for (auto &v : exPool16Cache)
      std::fill(v.begin(), v.end(), 0.0f);

    std::fill(scratchT0.begin(), scratchT0.end(), 0.0f);
    std::fill(scratchT1.begin(), scratchT1.end(), 0.0f);
    std::fill(scratchNewBranch.begin(), scratchNewBranch.end(), 0.0f);
    std::fill(scratchDirtyBranches.begin(), scratchDirtyBranches.end(), 0.0f);
    std::fill(scratchBranchDelta.begin(), scratchBranchDelta.end(), 0.0f);
    std::fill(scratchBypassDelta.begin(), scratchBypassDelta.end(), 0.0f);
    std::fill(scratchGproj.begin(), scratchGproj.end(), 0.0f);
    std::fill(scratchDeltaRelu.begin(), scratchDeltaRelu.end(), 0.0f);
    std::fill(scratchFlatDelta.begin(), scratchFlatDelta.end(), 0.0f);
    std::fill(scratchHidden.begin(), scratchHidden.end(), 0.0f);
    std::fill(oldGlobalV.begin(), oldGlobalV.end(), 0.0f);

    if (nThreads > 1)
      threadPool = std::make_unique<PersistentThreadPool>(nThreads - 1);
    else
      threadPool.reset();

    initialized = false;
  }

  void copy_weights_from(const MoEDoubleAccumulator &src) {
    branchConvLayers = src.branchConvLayers;
    nThreads = src.nThreads;
    minParallelDirtyHeads = src.minParallelDirtyHeads;
    minParallelActiveExperts = src.minParallelActiveExperts;
    denseDirtySqThreshold = src.denseDirtySqThreshold;
    expertPoolMode = src.expertPoolMode;
    routeSlowGlobals = src.routeSlowGlobals;

    ownedWeights = src.ownedWeights;
    weights = src.weights;

    reset_runtime_state();
  }

  void reset_profile() { profile = PhaseProfile{}; }

  template <typename Fn>
  void parallel_for_indices(int n, int min_parallel_n, Fn &&fn) {
    if (n <= 0)
      return;
    if (!threadPool || nThreads <= 1 || n < min_parallel_n) {
      for (int i = 0; i < n; ++i)
        fn(i);
      return;
    }
    threadPool->parallel_for(n, std::function<void(int)>(std::forward<Fn>(fn)));
  }

  template <typename Fn> void parallel_for_indices(int n, Fn &&fn) {
    parallel_for_indices(n, 2, std::forward<Fn>(fn));
  }

  long long total_weights() const {
    const auto &w = shared_weights();
    long long total = 0;
    auto add_vec = [&](const auto &v) { total += (long long)v.size(); };

    for (const auto &br : w.branches) {
      add_vec(br.l0.w);
      add_vec(br.l0.b);
      add_vec(br.l1.w);
      add_vec(br.l1.b);
      add_vec(br.l2.w);
      add_vec(br.l2.b);
    }

    add_vec(w.mixerWBr);
    add_vec(w.mixerWBp);
    add_vec(w.mixerB);
    add_vec(w.globalW);
    add_vec(w.globalB);
    add_vec(w.gateW);
    add_vec(w.gateB);

    for (const auto &ex : w.experts) {
      add_vec(ex.wConv);
      add_vec(ex.bConv);
      if (expertPoolMode == ExpertPoolMode::Gap)
        add_vec(ex.wHG);
      else if (expertPoolMode == ExpertPoolMode::Flat)
        add_vec(ex.wH);
      else
        add_vec(ex.wH16);
      add_vec(ex.bH);
      add_vec(ex.wWdl);
      add_vec(ex.bWdl);
    }

    return total;
  }

  long long single_expert_weights() const {
    const auto &w = shared_weights();
    if (w.experts.empty())
      return 0;

    long long total = 0;
    const Expert &ex = w.experts.front();
    auto add_vec = [&](const auto &v) { total += (long long)v.size(); };

    add_vec(ex.wConv);
    add_vec(ex.bConv);
    if (expertPoolMode == ExpertPoolMode::Gap)
      add_vec(ex.wHG);
    else if (expertPoolMode == ExpertPoolMode::Flat)
      add_vec(ex.wH);
    else
      add_vec(ex.wH16);
    add_vec(ex.bH);
    add_vec(ex.wWdl);
    add_vec(ex.bWdl);

    return total;
  }

  long long experts_total_weights() const {
    const auto &w = shared_weights();
    long long total = 0;
    auto add_vec = [&](const auto &v) { total += (long long)v.size(); };

    for (const auto &ex : w.experts) {
      add_vec(ex.wConv);
      add_vec(ex.bConv);
      if (expertPoolMode == ExpertPoolMode::Gap)
        add_vec(ex.wHG);
      else if (expertPoolMode == ExpertPoolMode::Flat)
        add_vec(ex.wH);
      else
        add_vec(ex.wH16);
      add_vec(ex.bH);
      add_vec(ex.wWdl);
      add_vec(ex.bWdl);
    }

    return total;
  }

  long long backbone_weights() const {
    return total_weights() - experts_total_weights();
  }

  long long runtime_topk_weights(int topk) const {
    const int k = std::clamp(topk, 0, nExperts);
    return backbone_weights() + (long long)k * single_expert_weights();
  }

  static void validate_fixed_architecture(const BenchConfig &cfg) {
    if (cfg.branchDim != bd || cfg.mixerOut != nf || cfg.nBypass != nBypass ||
        cfg.nGlobals != nGlobals || cfg.nExperts != nExperts ||
        cfg.expertBottleneck != ebo || cfg.expertHidden != eh) {
      throw std::runtime_error(
          "Weights dimensions do not match fixed native architecture");
    }
  }

  void init(const SharedMoEWeights *shared, const BenchConfig &cfg) {
    validate_fixed_architecture(cfg);

    if (!shared)
      throw std::runtime_error("Null shared MoE weights");

    branchConvLayers = cfg.branchConvLayers;
    nThreads = std::max(1, cfg.nThreads);
    minParallelDirtyHeads = std::max(1, cfg.minParallelDirtyHeads);
    minParallelActiveExperts = std::max(2, cfg.minParallelActiveExperts);
    denseDirtySqThreshold = std::clamp(cfg.denseDirtySqThreshold, 1, 64);
    expertPoolMode = cfg.expertPoolMode;
    routeSlowGlobals = cfg.routeSlowGlobals;

    const bool keep_owned = ownedWeights && (shared == ownedWeights.get());
    if (!keep_owned)
      ownedWeights.reset();
    weights = shared;
    reset_runtime_state();
  }

  void init(const BenchConfig &cfg) {
    auto &w = mutable_owned_weights();
    w.init_architecture(cfg.branchConvLayers);
    init(&w, cfg);
  }

  void fill_random(unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.05f);

    auto &w = mutable_owned_weights();
    w.init_architecture(branchConvLayers);

    auto fill = [&](auto &v) {
      for (float &x : v)
        x = nd(rng);
    };

    for (auto &br : w.branches) {
      fill(br.l0.w);
      fill(br.l0.b);
      if (branchConvLayers >= 2) {
        fill(br.l1.w);
        fill(br.l1.b);
      }
      if (branchConvLayers >= 3) {
        fill(br.l2.w);
        fill(br.l2.b);
      }
    }

    fill(w.mixerWBr);
    fill(w.mixerWBp);
    fill(w.mixerB);
    fill(w.globalW);
    fill(w.globalB);
    fill(w.gateW);
    fill(w.gateB);

    for (auto &ex : w.experts) {
      fill(ex.wConv);
      fill(ex.bConv);
      if (expertPoolMode == ExpertPoolMode::Gap) {
        fill(ex.wHG);
        for (int h = 0; h < eh; ++h) {
          for (int f = 0; f < ebo; ++f) {
            ex.wHGT[(size_t)f * eh + h] = ex.wHG[(size_t)h * ebo + f];
          }
        }
      } else if (expertPoolMode == ExpertPoolMode::Flat) {
        fill(ex.wH);
        for (int h = 0; h < eh; ++h) {
          for (int f = 0; f < ebo * 64; ++f) {
            ex.wHT[(size_t)f * eh + h] = ex.wH[(size_t)h * (ebo * 64) + f];
          }
        }
      } else {
        fill(ex.wH16);
        for (int h = 0; h < eh; ++h) {
          for (int f = 0; f < ebo * kPool2x2Regions; ++f) {
            ex.wH16T[(size_t)f * eh + h] =
                ex.wH16[(size_t)h * (ebo * kPool2x2Regions) + f];
          }
        }
      }
      fill(ex.bH);
      fill(ex.wWdl);
      fill(ex.bWdl);
    }
  }

  void branch_forward_bd16_fast(const Branch &br,
                                const float *HOT_RESTRICT in_planes,
                                float *HOT_RESTRICT out,
                                float *HOT_RESTRICT mid_plane,
                                float *HOT_RESTRICT l1_accum) {
#if defined(__ARM_NEON)
    conv3x3_relu_bd16_neon(in_planes, br.l0.w.data(), br.l0.b.data(), mid_plane,
                           br.l0.ic);
    depthwise_conv3x3_relu(mid_plane, br.l1.w.data(), br.l1.b.data(), l1_accum,
                           16);
    conv1x1_bd16_relu_neon(l1_accum, br.l2.w.data(), br.l2.b.data(), out);
#elif defined(__AVX2__)
    conv3x3_relu_bd16_dispatch(in_planes, br.l0.w.data(), br.l0.b.data(),
                               mid_plane, br.l0.ic);
    depthwise_conv3x3_relu(mid_plane, br.l1.w.data(), br.l1.b.data(), l1_accum,
                           16);
    conv1x1_bd16_relu_avx2(l1_accum, br.l2.w.data(), br.l2.b.data(), out);
#else
    conv3x3_relu_bd16_dispatch(in_planes, br.l0.w.data(), br.l0.b.data(),
                               mid_plane, br.l0.ic);
    depthwise_conv3x3_relu(mid_plane, br.l1.w.data(), br.l1.b.data(), l1_accum,
                           16);
    conv1x1_relu(l1_accum, br.l2.w.data(), br.l2.b.data(), out, 16, 16);
#endif
  }

  void branch_forward_with_scratch(int b, const float *in_planes, float *out,
                                   float *scratch0, float *scratch1) {
    const auto &br = shared_weights().branches[(size_t)b];

    if (branchConvLayers == 1) {
      if (bd == 16)
        conv3x3_relu_bd16_dispatch(in_planes, br.l0.w.data(), br.l0.b.data(),
                                   out, br.l0.ic);
      else
        conv3x3_relu(in_planes, br.l0.w.data(), br.l0.b.data(), out, br.l0.ic,
                     bd);
      return;
    }

    if (branchConvLayers == 2) {
      if (bd == 16) {
        conv3x3_relu_bd16_dispatch(in_planes, br.l0.w.data(), br.l0.b.data(),
                                   scratch0, br.l0.ic);
        conv3x3_relu(scratch0, br.l1.w.data(), br.l1.b.data(), out, 16, 16);
      } else {
        conv3x3_relu(in_planes, br.l0.w.data(), br.l0.b.data(), scratch0,
                     br.l0.ic, bd);
        conv3x3_relu(scratch0, br.l1.w.data(), br.l1.b.data(), out, bd, bd);
      }
      return;
    }

    if (bd == 16) {
      branch_forward_bd16_fast(br, in_planes, out, scratch0, scratch1);
      return;
    }
    conv3x3_relu(in_planes, br.l0.w.data(), br.l0.b.data(), scratch0, br.l0.ic,
                 bd);
    depthwise_conv3x3_relu(scratch0, br.l1.w.data(), br.l1.b.data(), scratch1,
                           bd);
    conv1x1_relu(scratch1, br.l2.w.data(), br.l2.b.data(), out, bd, bd);
  }

  void branch_forward(int b, const float *in_planes, float *out) {
    branch_forward_with_scratch(b, in_planes, out, scratchT0.data(),
                                scratchT1.data());
  }

  void rebuild_hidden_acc_from_flat(int e) {
    auto &hacc = hiddenAcc[(size_t)e];
    const auto &ex = shared_weights().experts[(size_t)e];
    const auto &flat = exReluCache[(size_t)e];
    const int flat_sz = ebo * 64;

    for (int out = 0; out < eh; ++out)
      hacc[(size_t)out] = ex.bH[(size_t)out];

    for (int i = 0; i < flat_sz; ++i) {
      const float fv = flat[(size_t)i];
      if (std::abs(fv) < 1e-7f)
        continue;
      const float *wt = &ex.wHT[(size_t)i * eh];
      for (int out = 0; out < eh; ++out)
        hacc[(size_t)out] += fv * wt[out];
    }
  }

  void rebuild_hidden_acc_from_gap(int e) {
    auto &hacc = hiddenAcc[(size_t)e];
    auto &gap = exGapCache[(size_t)e];
    const auto &ex = shared_weights().experts[(size_t)e];
    const auto &flat = exReluCache[(size_t)e];
    constexpr float inv64 = 1.0f / 64.0f;

    for (int bo = 0; bo < ebo; ++bo) {
      float s = 0.0f;
      const float *rbo = &flat[(size_t)bo * 64];
      FORCE_VECTORIZE
      for (int sq = 0; sq < 64; ++sq)
        s += rbo[sq];
      gap[(size_t)bo] = s * inv64;
    }

    for (int out = 0; out < eh; ++out)
      hacc[(size_t)out] = ex.bH[(size_t)out];

    for (int bo = 0; bo < ebo; ++bo) {
      const float gv = gap[(size_t)bo];
      if (std::abs(gv) < 1e-7f)
        continue;
      const float *wt = &ex.wHGT[(size_t)bo * eh];
      for (int out = 0; out < eh; ++out)
        hacc[(size_t)out] += gv * wt[out];
    }
  }

  void rebuild_hidden_acc_from_pool2x2(int e, bool max_pool) {
    auto &hacc = hiddenAcc[(size_t)e];
    auto &pool = exPool16Cache[(size_t)e];
    const auto &ex = shared_weights().experts[(size_t)e];
    const auto &flat = exReluCache[(size_t)e];

    for (int bo = 0; bo < ebo; ++bo) {
      const float *rbo = &flat[(size_t)bo * 64];
      for (int region = 0; region < kPool2x2Regions; ++region) {
        int r0 = 0, c0 = 0;
        pool2x2_region_base(region, r0, c0);
        const int sq0 = r0 * 8 + c0;
        const int sq1 = sq0 + 1;
        const int sq2 = sq0 + 8;
        const int sq3 = sq2 + 1;

        float pv;
        if (max_pool) {
          pv = std::max(std::max(rbo[sq0], rbo[sq1]),
                        std::max(rbo[sq2], rbo[sq3]));
        } else {
          pv = 0.25f * (rbo[sq0] + rbo[sq1] + rbo[sq2] + rbo[sq3]);
        }
        pool[(size_t)bo * kPool2x2Regions + region] = pv;
      }
    }

    for (int out = 0; out < eh; ++out)
      hacc[(size_t)out] = ex.bH[(size_t)out];

    const int pool_sz = ebo * kPool2x2Regions;
    for (int i = 0; i < pool_sz; ++i) {
      const float pv = pool[(size_t)i];
      if (std::abs(pv) < 1e-7f)
        continue;
      const float *wt = &ex.wH16T[(size_t)i * eh];
      for (int out = 0; out < eh; ++out)
        hacc[(size_t)out] += pv * wt[out];
    }
  }

  inline float global_proj_at(int oc, const float *g) const {
    const auto &shared = shared_weights();
    float s = shared.globalB[(size_t)oc];
    const float *w = &shared.globalW[(size_t)oc * nGlobals];
    for (int i = 0; i < nGlobals; ++i)
      s += w[i] * g[i];
    return s;
  }

  void top2_experts(const float *global, int &e0, int &e1, float &w0,
                    float &w1) const {
    e0 = 0;
    e1 = 1;
    float s0 = std::numeric_limits<float>::lowest();
    float s1 = std::numeric_limits<float>::lowest();
    const auto &shared = shared_weights();

    for (int e = 0; e < nExperts; ++e) {
      float s = shared.gateB[(size_t)e];
      const float *w = &shared.gateW[(size_t)e * nGlobals];
      if (routeSlowGlobals) {
        const int slow_n = std::min(nGlobals, 15);
        for (int i = 0; i < slow_n; ++i)
          s += w[i] * global[i];
      } else {
        for (int i = 0; i < nGlobals; ++i)
          s += w[i] * global[i];
      }

      if (s > s0) {
        s1 = s0;
        e1 = e0;
        s0 = s;
        e0 = e;
      } else if (s > s1) {
        s1 = s;
        e1 = e;
      }
    }

    const float m = std::max(s0, s1);
    const float p0 = std::exp(s0 - m);
    const float p1 = std::exp(s1 - m);
    const float z = p0 + p1;
    if (z > 1e-20f) {
      w0 = p0 / z;
      w1 = p1 / z;
    } else {
      w0 = 0.5f;
      w1 = 0.5f;
    }
  }

  void rebuild_expert_cache_from_mixer(int e) {
    auto &pre = exPreAccum[(size_t)e];
    auto &rel = exReluCache[(size_t)e];
    const auto &ex = shared_weights().experts[(size_t)e];
    for (int bo = 0; bo < ebo; ++bo) {
      float *pbo = &pre[(size_t)bo * 64];

      const float b = ex.bConv[(size_t)bo];
      for (int sq = 0; sq < 64; ++sq)
        pbo[sq] = b;

      const float *w = &ex.wConv[(size_t)bo * nf];
      for (int oc = 0; oc < nf; ++oc) {
        const float wc = w[oc];
        const float *in_plane = &mixerReluCache[(size_t)oc * 64];
#pragma GCC ivdep
        for (int sq = 0; sq < 64; ++sq)
          pbo[sq] += in_plane[sq] * wc;
      }

      float *rbo = &rel[(size_t)bo * 64];
      for (int sq = 0; sq < 64; ++sq)
        rbo[sq] = std::max(0.0f, pbo[sq]);
    }
    if (expertPoolMode == ExpertPoolMode::Gap)
      rebuild_hidden_acc_from_gap(e);
    else if (expertPoolMode == ExpertPoolMode::Flat)
      rebuild_hidden_acc_from_flat(e);
    else
      rebuild_hidden_acc_from_pool2x2(e, expertPoolMode ==
                                             ExpertPoolMode::Pool2x2Max);
    exValid[(size_t)e] = 1;
  }

  /**
   * @brief Performs a full forward pass of the model, discarding all cache.
   *
   * This is used for the very first evaluation of a game (where there is no
   * previous state to incrementally update from), or when the position changes
   * so drastically (e.g. >6 dirty branches) that an incremental update would be
   * slower than a full pass.
   *
   * @param inp The full spatial features of the current position.
   * @param active_experts Array containing the indices of the experts chosen by
   * the router.
   * @param active_count The number of active experts (usually 2).
   */
  void full_rebuild_accumulators(const FactorizedInput &inp,
                                 const int *active_experts, int active_count) {
    const auto &shared = shared_weights();
    const float *w_mixer_br = shared.mixerWBr.data();
    const float *w_mixer_bp = shared.mixerWBp.data();
    auto tBranch0 = Clock::now();
    // 1) Recompute and cache all branch outputs.
    parallel_for_indices(12, 2, [&](int b) {
      const float *pin = &inp.branches[b][0][0];
      float *bout = &branchCache[(size_t)b * bd * 64];
      branch_forward_with_scratch(b, pin, bout,
                                  scratchParallelBranch0[(size_t)b].data(),
                                  scratchParallelBranch1[(size_t)b].data());
    });
    auto tBranch1 = Clock::now();
    profile.fullBranchForwardUs += us(tBranch0, tBranch1);

    // 2) Rebuild mixer linear accumulator from branches + bypass + bias.
    auto tMix0 = Clock::now();
    // Bias initialize.
    for (int oc = 0; oc < nf; ++oc) {
      float *macc = &mixerLinearAccum[(size_t)oc * 64];
      const float bias = shared.mixerB[(size_t)oc];
      for (int sq = 0; sq < 64; ++sq)
        macc[sq] = bias;
    }

    // Branch+bypass accumulation using oc tiles to improve cache locality.
    const int ocTiles = (nf + kMixerOcTile - 1) / kMixerOcTile;
    parallel_for_indices(ocTiles, 2, [&](int tile) {
      const int oc0 = tile * kMixerOcTile;
      const int ocl = std::min(kMixerOcTile, nf - oc0);
      float *macc[kMixerOcTile];
      for (int t = 0; t < ocl; ++t)
        macc[t] = &mixerLinearAccum[(size_t)(oc0 + t) * 64];

      for (int b = 0; b < 12; ++b) {
        const float *bo = &branchCache[(size_t)b * bd * 64];
        for (int c = 0; c < bd; ++c) {
          const float *in_plane = &bo[(size_t)c * 64];
          for (int t = 0; t < ocl; ++t) {
            const float wc = w_mixer_br[((size_t)b * nf + (oc0 + t)) * bd + c];
            simd_add_scaled(macc[t], in_plane, wc, 64);
          }
        }
      }

      for (int bp = 0; bp < nBypass; ++bp) {
        const float *in_plane = &inp.bypass[bp][0];
        for (int t = 0; t < ocl; ++t) {
          const float wc = w_mixer_bp[(size_t)bp * nf + (oc0 + t)];
          simd_add_scaled(macc[t], in_plane, wc, 64);
        }
      }
    });
    auto tMix1 = Clock::now();
    profile.fullMixerAccumUs += us(tMix0, tMix1);

    // 3) Build post-ReLU mixer cache.
    auto tRelu0 = Clock::now();
    for (int oc = 0; oc < nf; ++oc) {
      const float g = global_proj_at(oc, inp.global);
      for (int sq = 0; sq < 64; ++sq) {
        const float v = mixerLinearAccum[(size_t)oc * 64 + sq] + g;
        mixerReluCache[(size_t)oc * 64 + sq] = std::max(0.0f, v);
      }
    }
    auto tRelu1 = Clock::now();
    profile.fullMixerReluUs += us(tRelu0, tRelu1);

    // 4) Build only active expert bottleneck caches (Top-2 routing).
    auto tEx0 = Clock::now();
    std::fill(exValid.begin(), exValid.end(), 0);
    if (active_count == 2 && active_experts[0] != active_experts[1]) {
      parallel_for_indices(2, 2, [&](int i) {
        rebuild_expert_cache_from_mixer(active_experts[i]);
      });
    } else {
      for (int i = 0; i < active_count; ++i)
        rebuild_expert_cache_from_mixer(active_experts[i]);
    }
    auto tEx1 = Clock::now();
    profile.fullExpertCacheUs += us(tEx0, tEx1);

    std::memcpy(oldGlobalV.data(), inp.global, sizeof(float) * nGlobals);

    initialized = true;
  }

  /**
   * @brief Performs an incremental network update by computing and applying
   * only the differences.
   *
   * This is the heart of the engine's speed. By passing a list of
   * `dirty_branches`, this function skips computing convolutions for pieces
   * that haven't moved. It calculates the delta for the branches that did
   * change, propagates that delta through the mixer layer, and updates the
   * bottlenecks of the active experts.
   *
   * @param cur The features for the new position.
   * @param prev The features for the old (parent) position.
   * @param dirty_branches Array of branch indices that changed.
   * @param dirty_count Number of branches that changed.
   * @param active_experts Array containing the indices of the currently active
   * experts.
   * @param active_count Number of active experts.
   */
  void update_incremental(const FactorizedInput &cur,
                          const FactorizedInput &prev,
                          const int *dirty_branches, int dirty_count,
                          const int *active_experts, int active_count) {
    const auto &shared = shared_weights();
    const float *w_mixer_br = shared.mixerWBr.data();
    const float *w_mixer_bp = shared.mixerWBp.data();
    if (!initialized) {
      full_rebuild_accumulators(cur, active_experts, active_count);
      return;
    }

    uint64_t dirty_sq_mask = 0ULL;
    std::fill(scratchDeltaRelu.begin(), scratchDeltaRelu.end(), 0.0f);
    std::fill(scratchFlatDelta.begin(), scratchFlatDelta.end(), 0.0f);

    // A) Branch deltas -> mixer linear accumulator.
    auto tA0 = Clock::now();
    parallel_for_indices(dirty_count, minParallelDirtyHeads, [&](int i) {
      const int b = dirty_branches[i];
      const float *pin = &cur.branches[b][0][0];
      float *new_cache = &scratchDirtyBranches[(size_t)i * (size_t)bd * 64];
      branch_forward_with_scratch(b, pin, new_cache,
                                  scratchParallelBranch0[(size_t)i].data(),
                                  scratchParallelBranch1[(size_t)i].data());
    });

    for (int i = 0; i < dirty_count; ++i) {
      const int b = dirty_branches[i];
      const float *new_branch =
          &scratchDirtyBranches[(size_t)i * (size_t)bd * 64];

      float *old_cache = &branchCache[(size_t)b * bd * 64];
      for (int c = 0; c < bd; ++c) {
        const float *old_plane = &old_cache[(size_t)c * 64];
        const float *new_plane = &new_branch[(size_t)c * 64];
        float *delta_plane = &scratchBranchDelta[(size_t)c * 64];

#pragma GCC ivdep
        for (int sq = 0; sq < 64; ++sq)
          delta_plane[sq] = new_plane[sq] - old_plane[sq];

        for (int sq = 0; sq < 64; ++sq) {
          if (std::abs(delta_plane[sq]) > 1e-9f)
            dirty_sq_mask |= (1ULL << sq);
        }
      }

      // Accumulate branch deltas across all 64 squares with SIMD-friendly
      // loops.
      constexpr int kTileOC = 8;
      const int ocTiles = (nf + kTileOC - 1) / kTileOC;
      for (int tile = 0; tile < ocTiles; ++tile) {
        const int oc0 = tile * kTileOC;
        const int ocl = std::min(kTileOC, nf - oc0);
        for (int c = 0; c < bd; ++c) {
          const float *ds = &scratchBranchDelta[(size_t)c * 64];
          for (int t = 0; t < ocl; ++t) {
            const float w = w_mixer_br[((size_t)b * nf + (oc0 + t)) * bd + c];
            float *macc = &mixerLinearAccum[(size_t)(oc0 + t) * 64];
            simd_add_scaled(macc, ds, w, 64);
          }
        }
      }

      std::memcpy(old_cache, new_branch, (size_t)bd * 64 * sizeof(float));
    }
    auto tA1 = Clock::now();
    profile.incBranchDeltaUs += us(tA0, tA1);

    // B) Bypass deltas -> mixer linear accumulator.
    auto tB0 = Clock::now();
    std::array<uint8_t, kMaxBypass> bypass_dirty{};
    for (int bp = 0; bp < nBypass; ++bp) {
      bool has_delta = false;
      for (int sq = 0; sq < 64; ++sq) {
        const float delta = cur.bypass[bp][sq] - prev.bypass[bp][sq];
        scratchBypassDelta[(size_t)bp * 64 + sq] = delta;
        if (std::abs(delta) > 1e-9f) {
          has_delta = true;
          dirty_sq_mask |= (1ULL << sq);
        }
      }
      bypass_dirty[(size_t)bp] = has_delta ? 1u : 0u;
    }

    // Accumulate only dirty bypass planes across all 64 squares.
    {
      constexpr int kTileOC = 8;
      const int ocTiles = (nf + kTileOC - 1) / kTileOC;
      for (int tile = 0; tile < ocTiles; ++tile) {
        const int oc0 = tile * kTileOC;
        const int ocl = std::min(kTileOC, nf - oc0);
        for (int bp = 0; bp < nBypass; ++bp) {
          if (!bypass_dirty[(size_t)bp])
            continue;
          const float *ds = &scratchBypassDelta[(size_t)bp * 64];
          for (int t = 0; t < ocl; ++t) {
            const float w = w_mixer_bp[(size_t)bp * nf + (oc0 + t)];
            float *macc = &mixerLinearAccum[(size_t)(oc0 + t) * 64];
            simd_add_scaled(macc, ds, w, 64);
          }
        }
      }
    }
    auto tB1 = Clock::now();
    profile.incBypassDeltaUs += us(tB0, tB1);

    // C) If globals changed, all squares are dirty (global is broadcast over
    // board).
    auto tCD0 = Clock::now();
    bool globals_changed = false;
    for (int g = 0; g < nGlobals; ++g) {
      if (std::abs(cur.global[g] - oldGlobalV[(size_t)g]) > 1e-6f) {
        globals_changed = true;
        break;
      }
    }
    auto tCD1 = Clock::now();
    profile.incGlobalReluUs += us(tCD0, tCD1);
    std::memcpy(oldGlobalV.data(), cur.global, sizeof(float) * nGlobals);
    if (globals_changed)
      dirty_sq_mask = ~0ULL;

    for (int oc = 0; oc < nf; ++oc)
      scratchGproj[(size_t)oc] = global_proj_at(oc, cur.global);

    // D) Compute post-ReLU deltas using ctz bit-scan over dirty squares.
    bool any_mixer_delta = false;
    uint64_t delta_sq_mask = 0ULL;
    for (int oc = 0; oc < nf; ++oc) {
      const float gb = scratchGproj[(size_t)oc];
      uint64_t mask = dirty_sq_mask;
      while (mask) {
        const int sq = __builtin_ctzll(mask);
        mask &= (mask - 1);

        const size_t idx = (size_t)oc * 64 + sq;
        const float oldv = mixerReluCache[idx];
        const float newv = std::max(0.0f, mixerLinearAccum[idx] + gb);
        const float d = newv - oldv;
        scratchDeltaRelu[idx] = d;
        mixerReluCache[idx] = newv;
        if (d != 0.0f) {
          any_mixer_delta = true;
          delta_sq_mask |= (1ULL << sq);
        }
      }
    }

    int dirty_sq_idx[64];
    int dirty_sq_count = 0;
    int dirty_region_idx[kPool2x2Regions];
    int dirty_region_count = 0;
    uint32_t region_mask = 0;
    uint64_t work = delta_sq_mask;
    while (work) {
      const int sq = __builtin_ctzll(work);
      work &= (work - 1);
      dirty_sq_idx[dirty_sq_count++] = sq;
      region_mask |= (1u << pool2x2_region_from_sq(sq));
    }
    const bool dense_dirty = dirty_sq_count >= denseDirtySqThreshold;

    while (region_mask) {
      const int region = __builtin_ctz(region_mask);
      region_mask &= (region_mask - 1);
      dirty_region_idx[dirty_region_count++] = region;
    }

    // D) Expert bottleneck deltas: update only active Top-2 expert caches.
    if (!any_mixer_delta) {
      std::array<double, kMaxExperts> reb_us{};
      parallel_for_indices(active_count, minParallelActiveExperts, [&](int i) {
        const int e = active_experts[i];
        if (!exValid[(size_t)e]) {
          const auto tReb0 = Clock::now();
          rebuild_expert_cache_from_mixer(e);
          const auto tReb1 = Clock::now();
          reb_us[(size_t)i] = us(tReb0, tReb1);
        }
      });
      for (int i = 0; i < active_count; ++i)
        profile.incExpertCacheRebuildUs += reb_us[(size_t)i];
      return;
    }

    std::array<uint8_t, kMaxExperts> new_valid{};
    std::array<int, kMaxExperts> processed_e{};
    std::array<double, kMaxExperts> reb_us{};
    std::array<double, kMaxExperts> bott_us{};
    std::array<double, kMaxExperts> hidden_us{};

    parallel_for_indices(active_count, minParallelActiveExperts, [&](int i) {
      const int e = active_experts[i];
      processed_e[(size_t)i] = e;

      if (!exValid[(size_t)e]) {
        const auto tReb0 = Clock::now();
        rebuild_expert_cache_from_mixer(e);
        const auto tReb1 = Clock::now();
        reb_us[(size_t)i] = us(tReb0, tReb1);
        return;
      }

      auto &pre = exPreAccum[(size_t)e];
      auto &rel = exReluCache[(size_t)e];
      auto &hacc = hiddenAcc[(size_t)e];
      auto &gap = exGapCache[(size_t)e];
      auto &pool16 = exPool16Cache[(size_t)e];
      const auto &ex = shared.experts[(size_t)e];
      const int flat_sz = ebo * 64;
      float *localFlatDelta = scratchParallelExpertDelta[(size_t)e].data();
      std::array<float, kMaxExpertBottleneck> localGapDelta{};
      std::array<float, (size_t)kMaxExpertBottleneck * kPool2x2Regions>
          localPoolDelta{};
      if (expertPoolMode == ExpertPoolMode::Flat)
        std::fill(localFlatDelta, localFlatDelta + flat_sz, 0.0f);

      const auto tBott0 = Clock::now();
      if (expertPoolMode == ExpertPoolMode::Gap) {
        constexpr float inv64 = 1.0f / 64.0f;
        for (int bo = 0; bo < ebo; ++bo) {
          float *pbo = &pre[(size_t)bo * 64];
          if (dense_dirty) {
            const float *w = &ex.wConv[(size_t)bo * nf];
            for (int ic = 0; ic < nf; ++ic) {
              const float wc = w[ic];
              const float *dr = &scratchDeltaRelu[(size_t)ic * 64];
              FORCE_VECTORIZE
              for (int sq = 0; sq < 64; ++sq)
                pbo[sq] += dr[sq] * wc;
            }
          } else {
            const float *w = &ex.wConv[(size_t)bo * nf];
            for (int k = 0; k < dirty_sq_count; ++k) {
              const int sq = dirty_sq_idx[k];
              float delta = 0.0f;
              for (int ic = 0; ic < nf; ++ic)
                delta += scratchDeltaRelu[(size_t)ic * 64 + sq] * w[ic];
              pbo[sq] += delta;
            }
          }

          float *rbo = &rel[(size_t)bo * 64];
          const float *pbo_ro = &pre[(size_t)bo * 64];
          float ch_sum_delta = 0.0f;
          if (dense_dirty) {
            for (int sq = 0; sq < 64; ++sq) {
              const float old_flat = rbo[sq];
              const float new_flat = std::max(0.0f, pbo_ro[sq]);
              rbo[sq] = new_flat;
              ch_sum_delta += (new_flat - old_flat);
            }
          } else {
            for (int k = 0; k < dirty_sq_count; ++k) {
              const int sq = dirty_sq_idx[k];
              const float old_flat = rbo[sq];
              const float new_flat = std::max(0.0f, pbo_ro[sq]);
              rbo[sq] = new_flat;
              ch_sum_delta += (new_flat - old_flat);
            }
          }
          const float gd = ch_sum_delta * inv64;
          localGapDelta[(size_t)bo] = gd;
          gap[(size_t)bo] += gd;
        }
      } else if (expertPoolMode == ExpertPoolMode::Flat) {
        for (int bo = 0; bo < ebo; ++bo) {
          float *pbo = &pre[(size_t)bo * 64];
          if (dense_dirty) {
            const float *w = &ex.wConv[(size_t)bo * nf];
            for (int ic = 0; ic < nf; ++ic) {
              const float wc = w[ic];
              const float *dr = &scratchDeltaRelu[(size_t)ic * 64];
              FORCE_VECTORIZE
              for (int sq = 0; sq < 64; ++sq)
                pbo[sq] += dr[sq] * wc;
            }
          } else {
            const float *w = &ex.wConv[(size_t)bo * nf];
            for (int k = 0; k < dirty_sq_count; ++k) {
              const int sq = dirty_sq_idx[k];
              float delta = 0.0f;
              for (int ic = 0; ic < nf; ++ic)
                delta += scratchDeltaRelu[(size_t)ic * 64 + sq] * w[ic];
              pbo[sq] += delta;
            }
          }

          float *rbo = &rel[(size_t)bo * 64];
          const float *pbo_ro = &pre[(size_t)bo * 64];
          if (dense_dirty) {
            for (int sq = 0; sq < 64; ++sq) {
              const float old_flat = rbo[sq];
              const float new_flat = std::max(0.0f, pbo_ro[sq]);
              rbo[sq] = new_flat;
              localFlatDelta[(size_t)bo * 64 + sq] = new_flat - old_flat;
            }
          } else {
            for (int k = 0; k < dirty_sq_count; ++k) {
              const int sq = dirty_sq_idx[k];
              const float old_flat = rbo[sq];
              const float new_flat = std::max(0.0f, pbo_ro[sq]);
              rbo[sq] = new_flat;
              localFlatDelta[(size_t)bo * 64 + sq] = new_flat - old_flat;
            }
          }
        }
      } else if (expertPoolMode == ExpertPoolMode::Pool2x2Avg) {
        constexpr float inv4 = 0.25f;
        for (int bo = 0; bo < ebo; ++bo) {
          float *pbo = &pre[(size_t)bo * 64];
          if (dense_dirty) {
            const float *w = &ex.wConv[(size_t)bo * nf];
            for (int ic = 0; ic < nf; ++ic) {
              const float wc = w[ic];
              const float *dr = &scratchDeltaRelu[(size_t)ic * 64];
              FORCE_VECTORIZE
              for (int sq = 0; sq < 64; ++sq)
                pbo[sq] += dr[sq] * wc;
            }
          } else {
            const float *w = &ex.wConv[(size_t)bo * nf];
            for (int k = 0; k < dirty_sq_count; ++k) {
              const int sq = dirty_sq_idx[k];
              float delta = 0.0f;
              for (int ic = 0; ic < nf; ++ic)
                delta += scratchDeltaRelu[(size_t)ic * 64 + sq] * w[ic];
              pbo[sq] += delta;
            }
          }

          float *rbo = &rel[(size_t)bo * 64];
          const float *pbo_ro = &pre[(size_t)bo * 64];
          if (dense_dirty) {
            for (int sq = 0; sq < 64; ++sq) {
              const float old_flat = rbo[sq];
              const float new_flat = std::max(0.0f, pbo_ro[sq]);
              rbo[sq] = new_flat;
              const int region = pool2x2_region_from_sq(sq);
              localPoolDelta[(size_t)bo * kPool2x2Regions + region] +=
                  (new_flat - old_flat) * inv4;
            }
          } else {
            for (int k = 0; k < dirty_sq_count; ++k) {
              const int sq = dirty_sq_idx[k];
              const float old_flat = rbo[sq];
              const float new_flat = std::max(0.0f, pbo_ro[sq]);
              rbo[sq] = new_flat;
              const int region = pool2x2_region_from_sq(sq);
              localPoolDelta[(size_t)bo * kPool2x2Regions + region] +=
                  (new_flat - old_flat) * inv4;
            }
          }

          const int regions_to_loop =
              dense_dirty ? kPool2x2Regions : dirty_region_count;
          for (int rk = 0; rk < regions_to_loop; ++rk) {
            const int region = dense_dirty ? rk : dirty_region_idx[rk];
            const size_t pidx = (size_t)bo * kPool2x2Regions + region;
            pool16[pidx] += localPoolDelta[pidx];
          }
        }
      } else {
        for (int bo = 0; bo < ebo; ++bo) {
          float *pbo = &pre[(size_t)bo * 64];
          float *rbo = &rel[(size_t)bo * 64];
          if (dense_dirty) {
            const float *w = &ex.wConv[(size_t)bo * nf];
            for (int ic = 0; ic < nf; ++ic) {
              const float wc = w[ic];
              const float *dr = &scratchDeltaRelu[(size_t)ic * 64];
              FORCE_VECTORIZE
              for (int sq = 0; sq < 64; ++sq)
                pbo[sq] += dr[sq] * wc;
            }
          } else {
            const float *w = &ex.wConv[(size_t)bo * nf];
            for (int k = 0; k < dirty_sq_count; ++k) {
              const int sq = dirty_sq_idx[k];
              float delta = 0.0f;
              for (int ic = 0; ic < nf; ++ic)
                delta += scratchDeltaRelu[(size_t)ic * 64 + sq] * w[ic];
              pbo[sq] += delta;
            }
          }

          if (dense_dirty) {
            for (int sq = 0; sq < 64; ++sq)
              rbo[sq] = std::max(0.0f, pbo[sq]);
          } else {
            for (int k = 0; k < dirty_sq_count; ++k) {
              const int sq = dirty_sq_idx[k];
              rbo[sq] = std::max(0.0f, pbo[sq]);
            }
          }

          const int regions_to_loop =
              dense_dirty ? kPool2x2Regions : dirty_region_count;
          for (int rk = 0; rk < regions_to_loop; ++rk) {
            const int region = dense_dirty ? rk : dirty_region_idx[rk];
            int r0 = 0, c0 = 0;
            pool2x2_region_base(region, r0, c0);
            const int sq0 = r0 * 8 + c0;
            const int sq1 = sq0 + 1;
            const int sq2 = sq0 + 8;
            const int sq3 = sq2 + 1;
            const float new_pool = std::max(std::max(rbo[sq0], rbo[sq1]),
                                            std::max(rbo[sq2], rbo[sq3]));
            const size_t pidx = (size_t)bo * kPool2x2Regions + region;
            const float old_pool = pool16[pidx];
            pool16[pidx] = new_pool;
            localPoolDelta[pidx] = new_pool - old_pool;
          }
        }
      }
      const auto tBott1 = Clock::now();
      bott_us[(size_t)i] = us(tBott0, tBott1);

      const auto tHidden0 = Clock::now();
      if (expertPoolMode == ExpertPoolMode::Gap) {
        for (int bo = 0; bo < ebo; ++bo) {
          const float delta = localGapDelta[(size_t)bo];
          if (std::abs(delta) < 1e-9f)
            continue;
          const float *wt = &ex.wHGT[(size_t)bo * eh];
          for (int out = 0; out < eh; ++out)
            hacc[(size_t)out] += delta * wt[out];
        }
      } else if (expertPoolMode == ExpertPoolMode::Flat) {
        if (dense_dirty) {
          for (int fi = 0; fi < flat_sz; ++fi) {
            const float delta = localFlatDelta[(size_t)fi];
            if (std::abs(delta) < 1e-7f)
              continue;
            const float *wt = &ex.wHT[(size_t)fi * eh];
#pragma GCC ivdep
            for (int out = 0; out < eh; ++out)
              hacc[(size_t)out] += delta * wt[out];
          }
        } else {
          for (int bo = 0; bo < ebo; ++bo) {
            for (int k = 0; k < dirty_sq_count; ++k) {
              const int sq = dirty_sq_idx[k];
              const int fi = bo * 64 + sq;
              const float delta = localFlatDelta[(size_t)fi];
              if (std::abs(delta) < 1e-7f)
                continue;
              const float *wt = &ex.wHT[(size_t)fi * eh];
              for (int out = 0; out < eh; ++out)
                hacc[(size_t)out] += delta * wt[out];
            }
          }
        }
      } else if (expertPoolMode == ExpertPoolMode::Pool2x2Avg) {
        const int regions_to_loop =
            dense_dirty ? kPool2x2Regions : dirty_region_count;
        for (int bo = 0; bo < ebo; ++bo) {
          for (int rk = 0; rk < regions_to_loop; ++rk) {
            const int region = dense_dirty ? rk : dirty_region_idx[rk];
            const int pidx = bo * kPool2x2Regions + region;
            const float delta = localPoolDelta[(size_t)pidx];
            if (std::abs(delta) < 1e-9f)
              continue;
            const float *wt = &ex.wH16T[(size_t)pidx * eh];
#pragma GCC ivdep
            for (int out = 0; out < eh; ++out)
              hacc[(size_t)out] += delta * wt[out];
          }
        }
      } else {
        const int regions_to_loop =
            dense_dirty ? kPool2x2Regions : dirty_region_count;
        for (int bo = 0; bo < ebo; ++bo) {
          for (int rk = 0; rk < regions_to_loop; ++rk) {
            const int region = dense_dirty ? rk : dirty_region_idx[rk];
            const int pidx = bo * kPool2x2Regions + region;
            const float delta = localPoolDelta[(size_t)pidx];
            if (std::abs(delta) < 1e-9f)
              continue;
            const float *wt = &ex.wH16T[(size_t)pidx * eh];
#pragma GCC ivdep
            for (int out = 0; out < eh; ++out)
              hacc[(size_t)out] += delta * wt[out];
          }
        }
      }
      const auto tHidden1 = Clock::now();
      hidden_us[(size_t)i] = us(tHidden0, tHidden1);
    });

    for (int i = 0; i < active_count; ++i) {
      profile.incExpertCacheRebuildUs += reb_us[(size_t)i];
      profile.incExpertBottleneckUs += bott_us[(size_t)i];
      profile.incHiddenDeltaUs += hidden_us[(size_t)i];
      new_valid[(size_t)processed_e[(size_t)i]] = 1;
    }
    exValid = new_valid;
  }

  /**
   * @brief Computes the final hidden layer and WDL output for a single expert.
   *
   * Takes the accumulated bottleneck state for the given expert, passes it
   * through the expert's hidden layer (with ReLU), and multiplies by the final
   * Win/Draw/Loss weights.
   *
   * @param e The index of the expert to run.
   * @param out_wdl A 3-element float array where the un-normalized WDL logits
   * will be stored.
   */
  void run_active_expert(int e, float out_wdl[3]) {
    out_wdl[0] = out_wdl[1] = out_wdl[2] = 0.0f;

    const auto &ex = shared_weights().experts[(size_t)e];
    const auto &hacc = hiddenAcc[(size_t)e];

    for (int h = 0; h < eh; ++h)
      scratchHidden[(size_t)h] = std::max(0.0f, hacc[(size_t)h]);

    float wdl_logits[3] = {ex.bWdl[0], ex.bWdl[1], ex.bWdl[2]};
    for (int o = 0; o < 3; ++o) {
      const float *w = &ex.wWdl[(size_t)o * eh];
      for (int h = 0; h < eh; ++h)
        wdl_logits[o] += scratchHidden[(size_t)h] * w[h];
    }

    const float mx =
        std::max(wdl_logits[0], std::max(wdl_logits[1], wdl_logits[2]));
    float s = 0.0f;
    float wdl_prob[3];
    for (int o = 0; o < 3; ++o) {
      wdl_prob[o] = std::exp(wdl_logits[o] - mx);
      s += wdl_prob[o];
    }
    if (s > 1e-20f) {
      const float inv = 1.0f / s;
      for (float &p : wdl_prob)
        p *= inv;
    }

    for (int o = 0; o < 3; ++o)
      out_wdl[o] = wdl_prob[o];
  }

  /**
   * @brief Combines the output of the top 2 routed experts based on their
   * routing weights.
   *
   * Evaluates both experts independently and takes a weighted average of their
   * Win/Draw/Loss logits using the probabilities assigned by the Router gate.
   *
   * @param e0 Index of the best expert.
   * @param e1 Index of the second-best expert.
   * @param w0 Routing weight (probability) for the best expert.
   * @param w1 Routing weight (probability) for the second-best expert.
   * @param out_wdl Array to store the combined WDL logits.
   */
  void run_top2_experts(int e0, int e1, float w0, float w1, float out_wdl[3]) {
    float wdl0[3] = {0.0f, 0.0f, 0.0f};
    float wdl1[3] = {0.0f, 0.0f, 0.0f};
    run_active_expert(e0, wdl0);
    run_active_expert(e1, wdl1);
    for (int o = 0; o < 3; ++o)
      out_wdl[o] = w0 * wdl0[o] + w1 * wdl1[o];
  }
};
