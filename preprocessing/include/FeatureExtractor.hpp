#pragma once
#include <array>
#include <cstdint>
#include "chess.hpp"

class FeatureExtractor {
public:
    static constexpr int NUM_SQUARES = 64;
    static constexpr int TOTAL_LAYERS = 42;

    typedef std::array<uint8_t, NUM_SQUARES> layer_t;

    typedef std::array<layer_t, NUM_SQUARES> chebyshev_table_t;
    typedef std::array<chebyshev_table_t, 2> chebyshev_table_by_side_t;
    typedef std::array<uint8_t, TOTAL_LAYERS * NUM_SQUARES> feature_array_t;

    

private:
    
    static chebyshev_table_by_side_t chebyshev_lookup;

    static chebyshev_table_t init_chebyshev(bool flip);


public:
    enum Layer {
        // Piece Presence (0-11)
        US_PAWN = 0, US_KNIGHT, US_BISHOP, US_ROOK, US_QUEEN, US_KING,
        THEM_PAWN, THEM_KNIGHT, THEM_BISHOP, THEM_ROOK, THEM_QUEEN, THEM_KING,

        // Legal Moves / Reachability (12-23)
        US_MOVE_PAWN, US_MOVE_KNIGHT, US_MOVE_BISHOP, US_MOVE_ROOK, US_MOVE_QUEEN, US_MOVE_KING,
        THEM_MOVE_PAWN, THEM_MOVE_KNIGHT, THEM_MOVE_BISHOP, THEM_MOVE_ROOK, THEM_MOVE_QUEEN, THEM_MOVE_KING,
        // Threats & Checks (24-27)
        US_THREATS, THEM_THREATS,
        US_CHECKS, THEM_CHECKS,

        // Defended (28-29)
        US_DEFENDED, THEM_DEFENDED,
        // King Distance (30-31)
        US_KING_DIST, THEM_KING_DIST,

        // Castling (32-35)
        US_CASTLE_K, US_CASTLE_Q, THEM_CASTLE_K, THEM_CASTLE_Q,

        // Material (36-41)
        US_MAT_PAWN, THEM_MAT_PAWN,
        US_MAT_MINOR, THEM_MAT_MINOR,
        US_MAT_MAJOR, THEM_MAT_MAJOR,
    };



    static feature_array_t board_to_features(const chess::Board& board);
};
