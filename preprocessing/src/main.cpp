#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include "chess.hpp"
#include "progressbar.hpp"
#include "FeatureExtractor.hpp"
#include "EvalNormalizer.hpp"
#include "Writers.hpp"

int main() {
    std::string fens_file = "/home/io/USI/io-bot/io-chess_engine/nnue/data/chessData.csv";
    // fens_file = "/home/io/USI/io-bot/io-chess_engine/nnue/preprocessing/test_fens.csv";
    
    // 1. Count lines for progress bar
    std::cout << "Counting lines in " << fens_file << "..." << std::endl;
    std::ifstream file_count(fens_file);
    if (!file_count.is_open()) {
        std::cerr << "Error: Could not open file " << fens_file << std::endl;
        return 1;
    }
    file_count.unsetf(std::ios_base::skipws);
    size_t total_lines = std::count(std::istreambuf_iterator<char>(file_count), 
                                    std::istreambuf_iterator<char>(), '\n');
    if (total_lines > 0) total_lines--; // Subtract header
    std::cout << "Found " << total_lines << " examples." << std::endl;

    // 2. Setup Writers
    const size_t FEATURES_PER_BOARD = 42*8*8; 
    const size_t BATCH_CAPACITY = 1000;
    DatasetWriter feature_writer("../data/processed/features.bin", FEATURES_PER_BOARD, BATCH_CAPACITY);
    LabelWriter label_writer("../data/processed/labels.bin", BATCH_CAPACITY);

    // 3. Process File
    std::ifstream file(fens_file);
    std::string line;
    std::getline(file, line); // Skip header

    ProgressBar pbar(total_lines, 50, false);
    size_t processed = 0;

    while (std::getline(file, line)) {
        // if (processed >= 100000) {
        //     break; // For testing, limit to 100k examples
        // }
        size_t comma_pos = line.find(',');
        if (comma_pos == std::string::npos) continue;

        std::string fen = line.substr(0, comma_pos);
        std::string eval_str = line.substr(comma_pos + 1);

        // Parse Board & Features
        chess::Board board(fen);
        std::array<uint8_t, FeatureExtractor::TOTAL_LAYERS * 64> features = FeatureExtractor::board_to_features(board);
        // Parse Eval
        float eval = EvalNormalizer::normalize(eval_str, board.sideToMove() == chess::Color::WHITE);

        // Write
        feature_writer.add(features);
        label_writer.add(eval);

        processed++;
        pbar.update(processed);
    }
    pbar.complete();

    return 0;
}
