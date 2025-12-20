#include "FeatureExtractor.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

// --- Static Member Initialization ---

FeatureExtractor::chebyshev_table_by_side_t FeatureExtractor::chebyshev_lookup = {
    FeatureExtractor::init_chebyshev(false),  // For White perspective
    FeatureExtractor::init_chebyshev(true)    // For Black perspective
};


// Initialize Chebyshev distance lookup table, mirror vertically if needed
FeatureExtractor::chebyshev_table_t FeatureExtractor::init_chebyshev(bool flip) {
    FeatureExtractor::chebyshev_table_t table;
    for (int sq1 = 0; sq1 < NUM_SQUARES; ++sq1) {
        for (int sq2 = 0; sq2 < NUM_SQUARES; ++sq2) {
            // Use raw coordinates. Do not flip rank here.
            int r1 = sq1 / 8, c1 = sq1 % 8;
            int r2 = sq2 / 8, c2 = sq2 % 8;
            int dist = std::max(std::abs(r1 - r2), std::abs(c1 - c2));
            table[sq1][sq2] = static_cast<uint8_t>(dist * 36);
        }
        if (flip) {
            // Flip vertically for black perspective
            std::array<uint8_t, NUM_SQUARES> flipped;
            for (int i = 0; i < NUM_SQUARES; ++i) {
                int rank = i / 8;
                int file = i % 8;
                int flipped_sq = (7 - rank) * 8 + file;
                flipped[i] = table[sq1][flipped_sq];
            }
            table[sq1] = flipped;            
        }
    }
    return table;
}

template <chess::PieceType::underlying PT>
constexpr int get_index() {
    if constexpr (PT == chess::PieceType::PAWN)   return 0;
    else if constexpr (PT == chess::PieceType::KNIGHT) return 1;
    else if constexpr (PT == chess::PieceType::BISHOP) return 2;
    else if constexpr (PT == chess::PieceType::ROOK)   return 3;
    else if constexpr (PT == chess::PieceType::QUEEN)  return 4;
    else if constexpr (PT == chess::PieceType::KING)   return 5;
}

// --- Feature Extraction Implementation ---
FeatureExtractor::feature_array_t FeatureExtractor::board_to_features(const chess::Board& board) {
    feature_array_t features = {0};

    // 1. Perspective Setup
    chess::Color us = board.sideToMove();
    chess::Color them = ~us;

    static constexpr int MAX_UINT8 = 255;



    // "Orient": If it's Black's turn, flip indices vertically (Rank 1 <-> Rank 8).
    // If White: mask is 56 (00111000 binary), which flips the ranks.
    // If Black: mask is 0, which leaves the square unchanged.
    const int orient_xor = (us == chess::Color::WHITE) ? 56 : 0;


    
    auto fill_layer = [&](int layer, chess::Bitboard bb) {

        // compute base pointer
        auto* layer_features = &features[layer * NUM_SQUARES];

        while (bb) {
            int raw_sq = bb.pop();

            // apply orientation
            int viewed_sq = raw_sq ^ orient_xor;

            layer_features[viewed_sq] = 1;
        }
    };

    auto fill_scalar = [&](int layer, uint8_t val) {
        int offset = layer * NUM_SQUARES;
        std::fill(features.begin() + offset, features.begin() + offset + NUM_SQUARES, val);
    };

    // 3. Board State & Caches
    auto us_pieces = board.us(us);
    auto them_pieces = board.us(them);
    auto occupied = board.occ();

    chess::Bitboard us_attacks_all = 0;
    chess::Bitboard them_attacks_all = 0;
    chess::Bitboard move_cache[12] = {0}; // [0-5]=Us Moves, [6-11]=Them Moves

    // --- Piece Presence (0-11) ---

    auto fill_presence = [&]<chess::PieceType::underlying PT>() {
        constexpr int i = get_index<PT>(); // Returns 0-5
        
        // Compiler can now optimize board.pieces() because PT is a constant
        fill_layer(US_PAWN + i, board.pieces(chess::PieceType(PT), us));
        fill_layer(THEM_PAWN + i, board.pieces(chess::PieceType(PT), them));
    };

    fill_presence.template operator()<chess::PieceType::PAWN>();
    fill_presence.template operator()<chess::PieceType::KNIGHT>();
    fill_presence.template operator()<chess::PieceType::BISHOP>();
    fill_presence.template operator()<chess::PieceType::ROOK>();
    fill_presence.template operator()<chess::PieceType::QUEEN>();
    fill_presence.template operator()<chess::PieceType::KING>();


    // --- PHASE 2: Legal Moves & Threats (12-25) ---
    chess::Bitboard ep_mask = 0;
    if (board.enpassantSq() != chess::Square::NO_SQ) {
        ep_mask = (1ULL << board.enpassantSq().index());
    }

    // Helper to generate moves for one side relative to perspective
    auto generate_moves = [&](chess::Color side, bool is_us) {
        int base_layer = is_us ? US_MOVE_PAWN : THEM_MOVE_PAWN;
        int cache_offset = is_us ? 0 : 6;
        chess::Bitboard& attack_acc = is_us ? us_attacks_all : them_attacks_all;
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
            chess::Bitboard valid_moves = push1 | push2 | (attacks & (enemies | ep_mask_effective));
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
        //    This works because chess::PieceType::KNIGHT is already of type 'underlying'
        process_piece.template operator()<chess::PieceType::KNIGHT>();
        process_piece.template operator()<chess::PieceType::BISHOP>();
        process_piece.template operator()<chess::PieceType::ROOK>();
        process_piece.template operator()<chess::PieceType::QUEEN>();
        process_piece.template operator()<chess::PieceType::KING>();

    };

    generate_moves(us, true);   // Us
    generate_moves(them, false); // Them

    // Threats
    fill_layer(US_THREATS, us_attacks_all & (them_pieces | ep_mask));
    fill_layer(THEM_THREATS, them_attacks_all & us_pieces); // Note: ep capture only possible by us

    // --- PHASE 3: Checks (26-27) ---
    chess::Square us_king = board.kingSq(us);
    chess::Square them_king = board.kingSq(them);

    auto get_checks = [&](chess::Color attacker, chess::Square target_king) -> chess::Bitboard {
        if (target_king == chess::Square::NO_SQ) return 0;

        bool is_us_attacking = (attacker == us);
        int cache_base = is_us_attacking ? 0 : 6;
        chess::Bitboard attacker_body = is_us_attacking ? us_pieces : them_pieces;

        // "Reverse" vision from King
        chess::Bitboard k_queen = chess::attacks::queen(target_king, occupied);
        chess::Bitboard k_rook = chess::attacks::rook(target_king, 0); 
        chess::Bitboard diag = k_queen & ~k_rook;
        chess::Bitboard orth = k_queen & k_rook;
        chess::Bitboard k_knight = chess::attacks::knight(target_king);
        
        // Find pawns that attack king (by checking where a pawn on king sq would attack)
        chess::Bitboard k_pawn = chess::attacks::pawn((attacker == chess::Color::WHITE ? chess::Color::BLACK : chess::Color::WHITE), target_king);

        chess::Bitboard checks = 0;
        checks |= move_cache[cache_base + 0] & k_pawn;   // Pawn
        checks |= move_cache[cache_base + 1] & k_knight; // Knight
        checks |= move_cache[cache_base + 2] & diag;     // Bishop
        checks |= move_cache[cache_base + 3] & orth;     // Rook
        checks |= move_cache[cache_base + 4] & (diag | orth); // Queen

        return checks & ~attacker_body;
    };

    fill_layer(US_CHECKS, get_checks(us, them_king));
    fill_layer(THEM_CHECKS, get_checks(them, us_king));

    // --- PHASE 4: Defended (28-29) ---
    fill_layer(US_DEFENDED, us_attacks_all & us_pieces);
    fill_layer(THEM_DEFENDED, them_attacks_all & them_pieces);

    // --- SCALING (Layers 0-29) ---
    for (int i = 0; i < 30 * NUM_SQUARES; ++i) {
        if (features[i] == 1) features[i] = MAX_UINT8;
    }

    // --- PHASE 5: King Distance (30-31) ---
    if (us_king != chess::Square::NO_SQ) {
        const auto& dists = chebyshev_lookup[(us == chess::Color::WHITE) ? 1 : 0][us_king.index()];
        std::copy(dists.begin(), dists.end(), features.begin() + US_KING_DIST * NUM_SQUARES);
    }
    if (them_king != chess::Square::NO_SQ) {
        const auto& dists = chebyshev_lookup[(them == chess::Color::WHITE) ? 0 : 1][them_king.index()];
        std::copy(dists.begin(), dists.end(), features.begin() + THEM_KING_DIST * NUM_SQUARES);
    }

    // --- PHASE 6: Castling (32-35) ---
    auto cr = board.castlingRights();
    using Side = chess::Board::CastlingRights::Side;

    if (cr.has(us, Side::KING_SIDE))    fill_scalar(US_CASTLE_K, MAX_UINT8);
    if (cr.has(us, Side::QUEEN_SIDE))   fill_scalar(US_CASTLE_Q, MAX_UINT8);
    if (cr.has(them, Side::KING_SIDE))  fill_scalar(THEM_CASTLE_K, MAX_UINT8);
    if (cr.has(them, Side::QUEEN_SIDE)) fill_scalar(THEM_CASTLE_Q, MAX_UINT8);

    // --- PHASE 7: Material (36-41) ---
    auto count = [&](chess::PieceType pt, chess::Color c) {
        return board.pieces(pt, c).count();
    };

    static constexpr int MAT_PAWN_SCALE = 31;
    static constexpr int MAT_MINOR_SCALE = 63;
    static constexpr int MAT_MAJOR_SCALE = 85;

    fill_scalar(US_MAT_PAWN, std::min(count(chess::PieceType::PAWN, us) * MAT_PAWN_SCALE, MAX_UINT8));
    fill_scalar(THEM_MAT_PAWN, std::min(count(chess::PieceType::PAWN, them) * MAT_PAWN_SCALE, MAX_UINT8));

    int us_min = count(chess::PieceType::KNIGHT, us) + count(chess::PieceType::BISHOP, us);
    int them_min = count(chess::PieceType::KNIGHT, them) + count(chess::PieceType::BISHOP, them);
    fill_scalar(US_MAT_MINOR, std::min(us_min * MAT_MINOR_SCALE, MAX_UINT8));
    fill_scalar(THEM_MAT_MINOR, std::min(them_min * MAT_MINOR_SCALE, MAX_UINT8));

    int us_maj = count(chess::PieceType::ROOK, us) + count(chess::PieceType::QUEEN, us);
    int them_maj = count(chess::PieceType::ROOK, them) + count(chess::PieceType::QUEEN, them);
    fill_scalar(US_MAT_MAJOR, std::min(us_maj * MAT_MAJOR_SCALE, MAX_UINT8));
    fill_scalar(THEM_MAT_MAJOR, std::min(them_maj * MAT_MAJOR_SCALE, MAX_UINT8));

    return features;
}
