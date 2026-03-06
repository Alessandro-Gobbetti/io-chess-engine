#pragma once

#include "../../../preprocessing/lib/chess/chess.hpp"
#include <memory>
#include <string>
#include <vector>

#pragma pack(push, 1)
struct PolyglotEntry {
  uint64_t key;
  uint16_t move;
  uint16_t weight;
  uint32_t learn;
};
#pragma pack(pop)

class PolyglotBook {
public:
  PolyglotBook(const std::string &book_path);
  ~PolyglotBook();

  bool load();
  chess::Move probe(const chess::Board &board,
                    const std::string &bookName = "Book");

private:
  std::string book_path_;
  std::vector<PolyglotEntry> entries_;

  uint64_t compute_polyglot_key(const chess::Board &board) const;
  chess::Move polyglot_to_move(uint16_t poly_move,
                               const chess::Board &board) const;
};
