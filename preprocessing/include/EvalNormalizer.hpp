/**
 * @file EvalNormalizer.hpp
 * @brief Utilities for normalizing chess evaluations into win probabilities.
 */
#pragma once
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>

class EvalNormalizer {
public:
    // Constants used for normalization
    static constexpr double MATE_SCORE = 20000.0;
    static constexpr double CLIP_LIMIT = 8000.0;
    static constexpr double GAMMA = 350.0;

    /**
     * Converts a raw CSV evaluation string (e.g., "35", "-100", "#5", "#-3")
     * into a normalized win probability [-1.0, 1.0].
     */
    static float normalize(const std::string& eval_str, bool is_white_to_move) {
        double cp_val = 0.0;

        // --- 1. Handle Mate Scores ("#5", "#-2") ---
        if (eval_str.find('#') != std::string::npos) {
            // Remove the '#' character to parse the number
            std::string moves_str = eval_str;
            moves_str.erase(std::remove(moves_str.begin(), moves_str.end(), '#'), moves_str.end());
            
            int moves = std::stoi(moves_str);
            
            int sign = (moves > 0) ? 1 : -1;
            cp_val = sign * (MATE_SCORE - (2000.0 * std::log(std::abs(moves))));
        } 
        // --- 2. Handle Centipawn Scores ("35", "-400") ---
        else {
            cp_val = std::stod(eval_str);
            
            // Clamp to keep range stable [-8000, 8000]
            cp_val = std::clamp(cp_val, -CLIP_LIMIT, CLIP_LIMIT);
        } 

        // --- 3. Apply Arctangent Normalization ---
        float normalized = static_cast<float>((2.0 / M_PI) * std::atan(cp_val / GAMMA));


        // --- 4. Adjust for Side to Move ---
        if (!is_white_to_move) {
            normalized = -normalized;
        }

        return normalized;
    }

    /**
     * Converts a normalized win probability [-1.0, 1.0] back to a raw CSV evaluation string.
     */
    static float to_centipawns(float normalized) {

        // Inverse of the arctangent normalization
        float cp_val = GAMMA * std::tan((normalized * M_PI) / 2.0f);
        // clamp to mate range
        cp_val = std::clamp(cp_val, static_cast<float>(-MATE_SCORE), static_cast<float>(MATE_SCORE));
        return cp_val;

    }
};
        