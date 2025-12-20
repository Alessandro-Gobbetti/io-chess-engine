#pragma once
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>

class EvalNormalizer {
public:
    // Constants used for normalization
    static constexpr double MATE_SCORE = 6000.0;
    static constexpr double CLIP_LIMIT = 4000.0;
    static constexpr double K = 0.00368208; // Win/Draw/Loss coefficient from Lichess

    /**
     * Converts a raw CSV evaluation string (e.g., "35", "-100", "#5", "#-3")
     * into a normalized win probability [-1.0, 1.0].
     */
    static float normalize(const std::string& eval_str, bool is_white_to_move) {
        double cp_val = 0.0;

        // --- 1. Handle Mate Scores ("#5", "#-2") ---
        if (eval_str.find('#') != std::string::npos) {
            try {
                // Remove the '#' character to parse the number
                std::string moves_str = eval_str;
                moves_str.erase(std::remove(moves_str.begin(), moves_str.end(), '#'), moves_str.end());
                
                int moves = std::stoi(moves_str);
                
                // Logic: Mate in 5 (+5) -> 6000 - 25 = 5975
                //        Mate in -5 (-5) -> -6000 + 25 = -5975
                int sign = (moves > 0) ? 1 : -1;
                cp_val = sign * (MATE_SCORE - std::abs(moves) * 5.0);
            } catch (...) {
                std::cerr << "Warning: Failed to parse mate: " << eval_str << std::endl;
                return 0.0f; // Fail safe
            }
        } 
        // --- 2. Handle Centipawn Scores ("35", "-400") ---
        else {
            try {
                cp_val = std::stod(eval_str);
                
                // Clamp to keep range stable [-3000, 3000]
                cp_val = std::clamp(cp_val, -CLIP_LIMIT, CLIP_LIMIT);
            } catch (...) {
                std::cerr << "Warning: Failed to parse centipawn: " << eval_str << std::endl;
                return 0.0f; // Fail safe
            }
        }

        // --- 3. Apply Sigmoid ---
        // Formula: 2 / (1 + e^(-k * cp)) - 1
        float normalized = static_cast<float>(2.0 / (1.0 + std::exp(-K * cp_val)) - 1.0);

        // --- 4. Adjust for Side to Move ---
        if (!is_white_to_move) {
            normalized = -normalized;
        }

        return normalized;
    }
};