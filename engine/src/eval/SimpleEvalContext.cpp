#include "SimpleEvalContext.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>

// =============================================================================
//  1. CONSTANTS & TUNABLE PARAMETERS
// =============================================================================

// --- Bitboard Masks ---
static constexpr uint64_t FILE_MASKS[8] = {
    0x0101010101010101ULL, 0x0202020202020202ULL, 0x0404040404040404ULL,
    0x0808080808080808ULL, 0x1010101010101010ULL, 0x2020202020202020ULL,
    0x4040404040404040ULL, 0x8080808080808080ULL};

static constexpr uint64_t RANK_MASKS[8] = {
    0x00000000000000FFULL, 0x000000000000FF00ULL, 0x0000000000FF0000ULL,
    0x00000000FF000000ULL, 0x000000FF00000000ULL, 0x0000FF0000000000ULL,
    0x00FF000000000000ULL, 0xFF00000000000000ULL};

// --- Material Values { Middlegame, Endgame } ---
const SimpleEvalContext::Score SimpleEvalContext::PIECE_VALUES[6] = {
    {100, 120}, // Pawn
    {320, 310}, // Knight
    {330, 320}, // Bishop
    {500, 525}, // Rook
    {975, 950}, // Queen
    {0, 0}      // King
};

// --- Mobility Bonus (Non-Linear) ---
static const SimpleEvalContext::Score MOBILITY_KNIGHT[9] = {
    {-20, -20}, {-10, -10}, {-5, -5}, {0, 0}, {5, 5}, {10, 10}, {15, 15}, {20, 20}, {25, 25}};

static const SimpleEvalContext::Score MOBILITY_BISHOP[14] = {
    {-10, -10}, {-5, -5}, {0, 0}, {5, 5}, {10, 10}, {15, 15}, {20, 20},
    {25, 25}, {30, 30}, {35, 35}, {35, 35}, {35, 35}, {35, 35}, {35, 35}};

static const SimpleEvalContext::Score MOBILITY_ROOK[15] = {
    {-10, -10}, {-5, -5}, {0, 0}, {5, 5}, {10, 10}, {15, 15}, {20, 20},
    {25, 25}, {30, 30}, {35, 35}, {40, 40}, {45, 45}, {50, 50}, {50, 50}, {50, 50}};

static const SimpleEvalContext::Score MOBILITY_QUEEN[28] = {
    {-10, -10}, {-5, -5}, {0, 0}, {0, 0}, {5, 5}, {5, 5}, {10, 10},
    {10, 10}, {15, 15}, {15, 15}, {20, 20}, {20, 20}, {25, 25}, {25, 25},
    {30, 30}, {30, 30}, {35, 35}, {35, 35}, {40, 40}, {40, 40}, {45, 45},
    {45, 45}, {50, 50}, {50, 50}, {50, 50}, {50, 50}, {50, 50}, {50, 50}};

// --- Strategic Bonuses ---
static const SimpleEvalContext::Score PASSED_PAWN_BONUS[8] = {
    {0, 0}, {5, 10}, {10, 20}, {20, 40}, {35, 70}, {60, 120}, {100, 200}, {0, 0}};

static const SimpleEvalContext::Score PASSED_PAWN_CONNECTED = {15, 30};

static const SimpleEvalContext::Score OUTPOST_BONUS = {25, 10};
static const SimpleEvalContext::Score ROOK_OPEN_FILE = {30, 15};
static const SimpleEvalContext::Score ROOK_SEMI_OPEN = {15, 7};
static const SimpleEvalContext::Score BISHOP_PAIR = {50, 50};
static const SimpleEvalContext::Score TEMPO_SCORE = {20, 10};

// --- Threats ---
static const SimpleEvalContext::Score THREAT_BY_PAWN = {-40, -40};
static const SimpleEvalContext::Score THREAT_MINOR_ON_MAJOR = {40, 20};
static const SimpleEvalContext::Score THREAT_ROOK_ON_QUEEN = {30, 15};

// --- King Safety & Endgame ---
static const SimpleEvalContext::Score SHIELD_MISSING = {-20, 0};
static const SimpleEvalContext::Score STORM_NEAR = {-30, -5};
static const SimpleEvalContext::Score STORM_FAR = {-10, 0};
static const int KING_PROXIMITY_WEIGHT = 10;

// --- PeSTO Tables ---
const SimpleEvalContext::Score SimpleEvalContext::PSQT[6][64] = {
    // 0: PAWN
    {{0, 0},    {0, 0},     {0, 0},   {0, 0},   {0, 0},    {0, 0},   {0, 0},
     {0, 0},    {98, 0},    {134, 0}, {61, 0},  {95, 0},   {68, 0},  {126, 0},
     {34, -10}, {-11, -10}, {-6, 0},  {7, -10}, {26, -10}, {31, 20}, {65, 20},
     {56, -10}, {25, -10},  {-20, 0}, {-14, 0}, {13, 0},   {6, 10},  {21, 20},
     {23, 25},  {12, 10},   {17, 0},  {-23, 0}, {-27, 0},  {-2, 0},  {-5, 10},
     {12, 20},  {17, 25},   {6, 10},  {10, 0},  {-25, 0},  {-26, 0}, {-4, 0},
     {-4, 0},   {-10, 0},   {3, 0},   {3, 0},   {33, 0},   {-12, 0}, {-35, 0},
     {-1, 0},   {-20, 0},   {-23, 0}, {-15, 0}, {24, 0},   {38, 0},  {-22, 0},
     {0, 0},    {0, 0},     {0, 0},   {0, 0},   {0, 0},    {0, 0},   {0, 0},
     {0, 0}},

    // 1: KNIGHT
    {{-167, -58}, {-89, -38},  {-34, -13},  {-49, -28}, {61, -31},  {-97, -27},
     {-15, -63},  {-107, -99}, {-73, -25},  {-41, -8},  {72, -25},  {36, -2},
     {23, -9},    {62, -25},   {7, -24},    {-17, -53}, {-47, -24}, {60, -20},
     {37, 10},    {65, 12},    {84, 15},    {129, 15},  {73, -10},  {44, -46},
     {-9, -17},   {17, 3},     {19, 22},    {53, 22},   {37, 22},   {69, 22},
     {18, 15},    {22, -25},   {-13, -35},  {4, -6},    {16, 16},   {13, 20},
     {28, 20},    {19, 13},    {21, 10},    {-8, -30},  {-23, -40}, {-9, -15},
     {12, -7},    {10, 5},     {19, 8},     {17, 10},   {25, -10},  {-16, -40},
     {-29, -26},  {-53, -25},  {-12, 4},    {-3, -6},   {-1, 3},    {18, -2},
     {-14, -20},  {-19, -26},  {-105, -60}, {-21, -55}, {-58, -30}, {-33, -20},
     {-17, -10},  {-28, -25},  {-19, -50},  {-23, -65}},

    // 2: BISHOP
    {{-29, -59}, {-4, -41},  {-8, -18},  {-14, -10}, {-14, -10}, {-8, -18},
     {-4, -41},  {-29, -59}, {-26, -37}, {8, -13},   {6, -7},    {10, -1},
     {10, -1},   {6, -7},    {8, -13},   {-26, -37}, {-11, -22}, {18, -6},
     {16, 6},    {21, 13},   {21, 13},   {16, 6},    {18, -6},   {-11, -22},
     {-7, -18},  {7, -2},    {5, 18},    {23, 18},   {23, 18},   {5, 18},
     {7, -2},    {-7, -18},  {-6, -20},  {5, -5},    {13, 17},   {24, 21},
     {24, 21},   {13, 17},   {5, -5},    {-6, -20},  {3, -25},   {10, -11},
     {12, 3},    {16, 10},   {16, 10},   {12, 3},    {10, -11},  {3, -25},
     {4, -25},   {13, -12},  {22, 2},    {17, 4},    {17, 4},    {22, 2},
     {13, -12},  {4, -25},   {-14, -60}, {-14, -50}, {-2, -24},  {-12, -18},
     {-12, -18}, {-2, -24},  {-14, -50}, {-14, -60}},

    // 3: ROOK
    {{32, 13},  {42, 10},  {32, 18}, {51, 15},  {56, 15},  {41, 18},  {38, 10},
     {36, 13},  {27, 9},   {32, 6},  {58, 19},  {62, 21},  {80, 21},  {55, 19},
     {29, 6},   {21, 9},   {-5, 7},  {19, 7},   {26, 22},  {36, 39},  {17, 39},
     {47, 22},  {11, 7},   {-7, 7},  {-12, 10}, {-10, -2}, {6, 13},   {24, 35},
     {29, 35},  {13, 13},  {-6, -2}, {-14, 10}, {-16, 10}, {-5, 3},   {1, 12},
     {11, 27},  {19, 27},  {4, 12},  {-6, 3},   {-19, 10}, {-20, 15}, {-10, 12},
     {2, 12},   {10, 23},  {11, 23}, {3, 12},   {-11, 12}, {-25, 15}, {-21, 15},
     {-10, 16}, {-3, 18},  {7, 19},  {2, 19},   {-2, 18},  {-8, 16},  {-24, 15},
     {-21, 15}, {-13, 16}, {-8, 18}, {6, 19},   {-1, 19},  {-8, 18},  {-16, 16},
     {-26, 15}},

    // 4: QUEEN
    {{-28, -69}, {0, -57},   {29, -47},  {12, -26},  {59, -26},  {44, -47},
     {43, -57},  {45, -69},  {-24, -55}, {-39, -31}, {-5, -22},  {1, -4},
     {-16, -4},  {57, -22},  {28, -31},  {54, -55},  {-13, -39}, {-17, -18},
     {18, 9},    {19, 29},   {46, 29},   {58, 9},    {13, -18},  {23, -39},
     {-19, -23}, {-18, -3},  {7, 21},    {30, 44},   {25, 44},   {16, 21},
     {-5, -3},   {-9, -23},  {-21, -29}, {-23, -13}, {-7, 24},   {29, 47},
     {47, 47},   {24, 24},   {-12, -13}, {-9, -29},  {-22, -38}, {-15, -16},
     {-5, -1},   {14, 14},   {15, 14},   {-6, -1},   {-14, -16}, {-6, -38},
     {-22, -44}, {-13, -24}, {-12, -10}, {9, 2},     {13, 2},    {9, -10},
     {-7, -24},  {-18, -44}, {-2, -55},  {-2, -43},  {1, -30},   {-2, -9},
     {-6, -9},   {10, -30},  {5, -43},   {4, -55}},

    // 5: KING
    {{-65, -200}, {23, -80},  {16, -60},  {-15, -40}, {-56, -40}, {-34, -60},
     {2, -80},    {13, -200}, {29, -80},  {-1, -40},  {-20, -20}, {-7, -20},
     {-8, -20},   {-4, -20},  {-38, -40}, {-29, -80}, {-9, -60},  {24, -20},
     {2, 0},      {-16, 20},  {-20, 20},  {6, 0},     {22, -20},  {-22, -60},
     {-17, -40},  {-20, -20}, {1, 20},    {12, 40},   {16, 40},   {5, 20},
     {-17, -20},  {-27, -40}, {-49, -40}, {-1, -20},  {27, 20},   {39, 40},
     {46, 40},    {48, 20},   {32, -20},  {-51, -40}, {-14, -60}, {3, -20},
     {22, 0},     {29, 20},   {44, 20},   {29, 0},    {30, -20},  {-16, -60},
     {-55, -80},  {-10, -40}, {-9, -20},  {-19, -20}, {-11, -20}, {-2, -20},
     {-15, -40},  {-61, -80}, {0, -200},  {-54, -80}, {-33, -60}, {-35, -40},
     {-35, -40},  {-33, -60}, {-53, -80}, {-16, -200}}};

// =============================================================================
//  2. HELPERS
// =============================================================================

// Helper: Get mirror square
static inline int mirror(int sq) {
    return sq ^ 56;
}

// =============================================================================
//  3. THE NEW EVALUATE FUNCTION
// =============================================================================

float SimpleEvalContext::evaluate(const Board &board) {
    Score score = {0, 0};
    int phase = 0;
    const int PHASE_TOTAL = 24;

    Color us = board.sideToMove();
    Color them = ~us;

    Bitboard usPawns = board.pieces(PieceType::PAWN, us);
    Bitboard themPawns = board.pieces(PieceType::PAWN, them);
    Bitboard occupied = board.occ();

    // -------------------------------------------------------------------------
    //  LAMBDA: EVALUATE ONE SIDE
    // -------------------------------------------------------------------------
    auto evalSide = [&](Color side, Color enemy, Bitboard myPawns, Bitboard enemyPawns, int& phaseCount) -> Score {
        Score s = {0, 0};
        bool isWhite = (side == Color::WHITE);
        
        Bitboard myMinorAttacks = 0;
        Bitboard myRookAttacks = 0;
        
        // --- 1. PAWNS (Material + PSQT + Passed Logic) ---
        Bitboard p = myPawns;
        while (p) {
            int sq = p.pop();
            int psqtIdx = isWhite ? sq : mirror(sq);
            s += PIECE_VALUES[0] + PSQT[0][psqtIdx];
            
            // Passed Pawn Recognition
            int file = sq % 8;
            int rank = sq / 8;
            
            Bitboard forwardSpan = FILE_MASKS[file];
            if (file > 0) forwardSpan |= FILE_MASKS[file - 1];
            if (file < 7) forwardSpan |= FILE_MASKS[file + 1];

            if (isWhite) {
                forwardSpan &= ~((1ULL << (sq + 1)) - 1); // clears 0..sq
                forwardSpan &= ~RANK_MASKS[rank]; 
            } else {
                forwardSpan &= ((1ULL << sq) - 1); 
            }

            if (!(forwardSpan & enemyPawns)) {
                int relRank = isWhite ? rank : 7 - rank;
                s += PASSED_PAWN_BONUS[relRank];
                
                // Connected?
                Bitboard supportMask = 0;
                if (isWhite) {
                    if (file > 0) supportMask |= (1ULL << (sq - 9));
                    if (file < 7) supportMask |= (1ULL << (sq - 7));
                } else {
                    if (file > 0) supportMask |= (1ULL << (sq + 7));
                    if (file < 7) supportMask |= (1ULL << (sq + 9));
                }
                
                if (supportMask & myPawns) {
                    s += PASSED_PAWN_CONNECTED;
                }
            }
        }

        // --- 2. KNIGHTS ---
        Bitboard n = board.pieces(PieceType::KNIGHT, side);
        while (n) {
            int sq = n.pop();
            int psqtIdx = isWhite ? sq : mirror(sq);
            s += PIECE_VALUES[1] + PSQT[1][psqtIdx];
            phaseCount += 1;

            Bitboard attacks = chess::attacks::knight(sq);
            myMinorAttacks |= attacks;
            
            int mobCount = (attacks & ~board.us(side)).count();
            s += MOBILITY_KNIGHT[std::min(mobCount, 8)];

            // Outposts
            int rank = sq / 8;
            int relRank = isWhite ? rank : 7 - rank;
            if (relRank >= 3 && relRank <= 5) {
                Bitboard pawnSupport = 0;
                if (isWhite) {
                    if (sq % 8 != 0) pawnSupport |= (1ULL << (sq - 9));
                    if (sq % 8 != 7) pawnSupport |= (1ULL << (sq - 7));
                } else {
                    if (sq % 8 != 0) pawnSupport |= (1ULL << (sq + 7));
                    if (sq % 8 != 7) pawnSupport |= (1ULL << (sq + 9));
                }
                if ((pawnSupport & myPawns)) {
                     // Check if safe from enemy pawns (single square check)
                     Bitboard enemyPawnAttacks = chess::attacks::pawn(enemy, Square(sq)); 
                     if (!(enemyPawnAttacks & enemyPawns)) {
                         s += OUTPOST_BONUS;
                     }
                }
            }
        }

        // --- 3. BISHOPS ---
        Bitboard b = board.pieces(PieceType::BISHOP, side);
        if (b.count() >= 2) s += BISHOP_PAIR;
        while (b) {
            int sq = b.pop();
            int psqtIdx = isWhite ? sq : mirror(sq);
            s += PIECE_VALUES[2] + PSQT[2][psqtIdx];
            phaseCount += 1;

            Bitboard attacks = chess::attacks::bishop(sq, occupied);
            myMinorAttacks |= attacks;
            int mobCount = (attacks & ~board.us(side)).count();
            s += MOBILITY_BISHOP[std::min(mobCount, 13)];
        }

        // --- 4. ROOKS ---
        Bitboard r = board.pieces(PieceType::ROOK, side);
        while (r) {
            int sq = r.pop();
            int psqtIdx = isWhite ? sq : mirror(sq);
            s += PIECE_VALUES[3] + PSQT[3][psqtIdx];
            phaseCount += 2;

            Bitboard attacks = chess::attacks::rook(sq, occupied);
            myRookAttacks |= attacks;
            int mobCount = (attacks & ~board.us(side)).count();
            s += MOBILITY_ROOK[std::min(mobCount, 14)];

            // Open Files
            int file = sq % 8;
            Bitboard fileMask = FILE_MASKS[file];
            if (!(myPawns & fileMask)) {
                if (!(enemyPawns & fileMask)) s += ROOK_OPEN_FILE;
                else s += ROOK_SEMI_OPEN;
            }
        }

        // --- 5. QUEENS ---
        Bitboard q = board.pieces(PieceType::QUEEN, side);
        while (q) {
            int sq = q.pop();
            int psqtIdx = isWhite ? sq : mirror(sq);
            s += PIECE_VALUES[4] + PSQT[4][psqtIdx];
            phaseCount += 4;

            Bitboard attacks = chess::attacks::queen(sq, occupied);
            int mobCount = (attacks & ~board.us(side)).count();
            s += MOBILITY_QUEEN[std::min(mobCount, 27)];
        }
        
        // --- 6. KING SAFETY ---
        int kSq = board.kingSq(side).index();
        s += PIECE_VALUES[5] + PSQT[5][isWhite ? kSq : mirror(kSq)];

        int kRank = kSq / 8;
        int kFile = kSq % 8;
        bool checkSafety = isWhite ? (kRank <= 2) : (kRank >= 5);

        if (checkSafety) {
            int startF = std::max(0, kFile - 1);
            int endF = std::min(7, kFile + 1);
            for (int f = startF; f <= endF; ++f) {
                Bitboard fileMask = FILE_MASKS[f];
                // Shield
                Bitboard shieldMask;
                if (isWhite) shieldMask = fileMask & (RANK_MASKS[1] | RANK_MASKS[2]);
                else         shieldMask = fileMask & (RANK_MASKS[6] | RANK_MASKS[5]);
                if (!(myPawns & shieldMask)) s += SHIELD_MISSING; 

                // Storm
                Bitboard stormMask;
                if (isWhite) stormMask = fileMask & (RANK_MASKS[3] | RANK_MASKS[4] | RANK_MASKS[5]);
                else         stormMask = fileMask & (RANK_MASKS[4] | RANK_MASKS[3] | RANK_MASKS[2]);
                if (enemyPawns & stormMask) s += STORM_NEAR;
                else {
                    if (isWhite) stormMask = fileMask & (RANK_MASKS[6]);
                    else         stormMask = fileMask & (RANK_MASKS[1]);
                    if (enemyPawns & stormMask) s += STORM_FAR;
                }
            }
        }

        // --- 7. TACTICAL THREATS ---
        // A. Enemy Pawn attacking my Non-Pawn (Penalty)
        // Generate squares attacked by ALL enemy pawns efficiently via shifts
        Bitboard enemyPawnAttacks;
        if (enemy == Color::WHITE) {
            // White captures NorthEast (+9) and NorthWest (+7)
            enemyPawnAttacks = ((enemyPawns << 9) & ~FILE_MASKS[0]) | 
                               ((enemyPawns << 7) & ~FILE_MASKS[7]);
        } else {
            // Black captures SouthEast (-7) and SouthWest (-9)
            // Note: Directions are relative to index (0=A1). 
            // -9 is A8->H7 (down-right? no, 56-9=47=H6). -9 needs not file A?
            // H8(63) - 9 = 54(G7). A8(56)-9=47(H6).
            // -9 is effectively South-East (visually South-West from black perspective).
            // -7 is effectively South-West (visually South-East from black perspective).
            enemyPawnAttacks = ((enemyPawns >> 9) & ~FILE_MASKS[7]) | 
                               ((enemyPawns >> 7) & ~FILE_MASKS[0]);
        }

        // My pieces that are NOT pawns
        Bitboard myValuablePieces = board.us(side) & ~myPawns;
        if (enemyPawnAttacks & myValuablePieces) {
             int threatCount = (enemyPawnAttacks & myValuablePieces).count();
             Score penalty = { THREAT_BY_PAWN.mg * threatCount, THREAT_BY_PAWN.eg * threatCount };
             s += penalty;
        }

        // B. My Minor attacking Enemy Major (Bonus)
        Bitboard enemyMajor = board.pieces(PieceType::ROOK, enemy) | board.pieces(PieceType::QUEEN, enemy);
        if (myMinorAttacks & enemyMajor) {
             int threatCount = (myMinorAttacks & enemyMajor).count();
             Score bonus = { THREAT_MINOR_ON_MAJOR.mg * threatCount, THREAT_MINOR_ON_MAJOR.eg * threatCount };
             s += bonus;
        }
        
        // C. My Rook attacking Enemy Queen (Bonus)
        Bitboard enemyQueen = board.pieces(PieceType::QUEEN, enemy);
        if (myRookAttacks & enemyQueen) {
             int threatCount = (myRookAttacks & enemyQueen).count();
             Score bonus = { THREAT_ROOK_ON_QUEEN.mg * threatCount, THREAT_ROOK_ON_QUEEN.eg * threatCount };
             s += bonus;
        }

        return s;
    };

    // Calculate Scores
    Score scoreUs = evalSide(us, them, usPawns, themPawns, phase);
    Score scoreThem = evalSide(them, us, themPawns, usPawns, phase);

    score += scoreUs;
    score -= scoreThem;

    // -------------------------------------------------------------------------
    //  PHASE INTERPOLATION & ENDGAME LOGIC
    // -------------------------------------------------------------------------
    int p = std::clamp(phase, 0, PHASE_TOTAL);

    // --- 8. KING PROXIMITY (Endgame Only) ---
    // If we are in the endgame (p is low), we want the king to be close to pawns.
    if (p < 10) {
        // Evaluate Us
        int usKingSq = board.kingSq(us).index();
        Bitboard allPawns = usPawns | themPawns;
        if (allPawns) {
            int minDist = 99;
            Bitboard tempP = allPawns;
            while (tempP) {
                int pSq = tempP.pop();
                int dist = std::max(std::abs((usKingSq/8) - (pSq/8)), std::abs((usKingSq%8) - (pSq%8)));
                if (dist < minDist) minDist = dist;
            }
            score.eg += (7 - minDist) * KING_PROXIMITY_WEIGHT;
        }
        
        // Evaluate Them (Symmetric subtraction)
        int themKingSq = board.kingSq(them).index();
        if (allPawns) {
            int minDist = 99;
            Bitboard tempP = allPawns;
            while (tempP) {
                int pSq = tempP.pop();
                int dist = std::max(std::abs((themKingSq/8) - (pSq/8)), std::abs((themKingSq%8) - (pSq%8)));
                if (dist < minDist) minDist = dist;
            }
            score.eg -= (7 - minDist) * KING_PROXIMITY_WEIGHT;
        }
    }

    // Interpolate: 24 = Full Middlegame, 0 = Full Endgame
    int finalScore = (score.mg * p + score.eg * (PHASE_TOTAL - p)) / PHASE_TOTAL;

    // Add Tempo
    finalScore += TEMPO_SCORE.mg; 

    return static_cast<float>(finalScore);
}
