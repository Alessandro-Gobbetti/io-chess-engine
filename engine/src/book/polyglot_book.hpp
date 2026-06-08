#pragma once

/**
 * @file polyglot_book.hpp
 * @brief Polyglot opening book parser and probe utility.
 *
 * Provides a loader and prober for standard Polyglot (.bin) opening books,
 * returning the best move weighted by frequency.
 */

#include "../../../preprocessing/lib/chess/chess.hpp"
#include <memory>
#include <string>
#include <vector>

#pragma pack(push, 1)
/**
 * @struct PolyglotEntry
 * @brief Represents a single entry in a Polyglot book file.
 */
struct PolyglotEntry {
  uint64_t key;     ///< Zobrist hash of the position.
  uint16_t move;    ///< Encoded move in Polyglot format.
  uint16_t weight;  ///< Weight or frequency of the move.
  uint32_t learn;   ///< Learn score (often unused).
};
#pragma pack(pop)

/**
 * @class PolyglotBook
 * @brief Manages loading and probing of Polyglot opening books.
 */
class PolyglotBook {
public:
  /**
   * @brief Constructs a PolyglotBook instance.
   * 
   * @param book_path File path to the Polyglot (.bin) file.
   */
  PolyglotBook(const std::string &book_path);
  ~PolyglotBook();

  /**
   * @brief Loads the book into memory.
   * 
   * @return true if successful, false if the file could not be read.
   */
  bool load();
  
  /**
   * @brief Probes the book for a move matching the current board state.
   * 
   * Moves are selected probabilistically based on their relative weights.
   * 
   * @param board The chess board to probe.
   * @param bookName Optional name for logging purposes.
   * @return The selected move, or NO_MOVE if no match is found.
   */
  chess::Move probe(const chess::Board &board,
                    const std::string &bookName = "Book");

private:
  std::string book_path_;
  std::vector<PolyglotEntry> entries_;

  uint64_t compute_polyglot_key(const chess::Board &board) const;
  chess::Move polyglot_to_move(uint16_t poly_move,
                               const chess::Board &board) const;
};
