#pragma once

/**
 * @file TT.h
 * @brief Transposition Table implementation using a 3+1 Cluster Architecture.
 *
 * Implements a lockless Transposition Table (TT) utilizing 64-byte cache-line
 * aligned clusters. Each cluster contains 3 search entries and 1 static 
 * evaluation entry. Data integrity is ensured using XOR checksum signatures.
 */

#include "../Types.h"
#include <atomic>
#include <cstdint>
#include <optional>

/**
 * @enum Bound
 * @brief Represents the type of score stored in the Transposition Table.
 */
enum class Bound : uint8_t {
  NONE = 0,      ///< Invalid or empty bound.
  UPPER = 1,     ///< Score is an upper bound (failed low).
  LOWER = 2,     ///< Score is a lower bound (failed high).
  EXACT = 3,     ///< Score is exact (from a PV node).
  LAZY_MASK = 4  ///< Flag indicating the entry was populated via lazy evaluation.
};

/**
 * @struct TTEntry
 * @brief 16-byte Transposition Table entry for search results.
 *
 * Uses a lockless XOR signature approach to detect data tearing during concurrent
 * reads and writes. Layout: [Move:16][Score:16][StaticEval:16][Depth:8][Bound:3][Age:5] = 64 bits.
 */
struct TTEntry {
  uint64_t data;      ///< Packed search data.
  uint64_t signature; ///< Checksum signature (Key ^ Data).

  /**
   * @brief Probes the entry to safely retrieve data if the key matches.
   * 
   * @param key The 64-bit Zobrist key of the position.
   * @param out_data Output parameter for the packed data if successful.
   * @return True if the entry belongs to the key and is not torn.
   */
  bool probe(uint64_t key, uint64_t &out_data) const {
    uint64_t d = __atomic_load_n(&data, __ATOMIC_RELAXED);
    uint64_t s = __atomic_load_n(&signature, __ATOMIC_RELAXED);
    if ((s ^ d) == key) {
      out_data = d;
      return true;
    }
    return false;
  }

  /**
   * @brief Atomically stores search data into the entry.
   * 
   * @param key 64-bit Zobrist key.
   * @param score Minimax score.
   * @param sEval Static evaluation score.
   * @param depth Remaining search depth.
   * @param bound Score bound type.
   * @param move Best move found.
   * @param age Search generation age.
   */
  void save(uint64_t key, int16_t score, int16_t sEval, uint8_t depth,
            Bound bound, uint16_t move, uint8_t age) {
    uint64_t newData =
        (static_cast<uint64_t>(move) << 48) |
        (static_cast<uint64_t>(static_cast<uint16_t>(score)) << 32) |
        (static_cast<uint64_t>(static_cast<uint16_t>(sEval)) << 16) |
        (static_cast<uint64_t>(depth) << 8) |
        (static_cast<uint64_t>(static_cast<uint8_t>(bound) & 0x7) << 5) |
        (age & 0x1F);

    __atomic_store_n(&signature, key ^ newData, __ATOMIC_RELAXED);
    __atomic_store_n(&data, newData, __ATOMIC_RELAXED);
  }

  // Helpers for extracting fields
  static uint16_t getMove(uint64_t d) { return static_cast<uint16_t>(d >> 48); }
  static int16_t getScore(uint64_t d) { return static_cast<int16_t>((d >> 32) & 0xFFFF); }
  static int16_t getStaticEval(uint64_t d) { return static_cast<int16_t>((d >> 16) & 0xFFFF); }
  static uint8_t getDepth(uint64_t d) { return static_cast<uint8_t>((d >> 8) & 0xFF); }
  static Bound getBound(uint64_t d) { return static_cast<Bound>((d >> 5) & 0x7); }
  static uint8_t getAge(uint64_t d) { return static_cast<uint8_t>(d & 0x1F); }
};

/**
 * @struct EvalEntry
 * @brief 16-byte entry dedicated exclusively to caching static evaluations.
 *
 * Helps offload the expensive neural network evaluations without evicting
 * valuable search trees. Uses the same XOR checksum mechanism.
 */
struct alignas(16) EvalEntry {
  std::atomic<uint64_t> payload;   ///< Packed static evaluation score.
  std::atomic<uint64_t> signature; ///< Checksum signature.

  /**
   * @brief Stores the static evaluation atomically.
   */
  void store(uint64_t key, int16_t score) {
    uint64_t p = static_cast<uint64_t>(static_cast<uint16_t>(score));
    uint64_t s = key ^ p;
    signature.store(s, std::memory_order_relaxed);
    payload.store(p, std::memory_order_relaxed);
  }

  /**
   * @brief Probes for a cached static evaluation.
   */
  bool probe(uint64_t key, int16_t &score) const {
    uint64_t p = payload.load(std::memory_order_relaxed);
    uint64_t s = signature.load(std::memory_order_relaxed);
    if ((s ^ p) == key) {
      score = static_cast<int16_t>(p & 0xFFFF);
      return true;
    }
    return false;
  }
};

/**
 * @struct Cluster
 * @brief 64-byte cache-line aligned structure holding TT entries.
 *
 * Contains 3 `TTEntry` search slots and 1 `EvalEntry` slot.
 */
struct alignas(64) Cluster {
  TTEntry entries[3]; ///< Search entries.
  EvalEntry eval;     ///< Dedicated static evaluation entry.
};
static_assert(sizeof(Cluster) == 64, "Cluster must be exactly 64 bytes");

/**
 * @class TranspositionTable
 * @brief The main Transposition Table.
 *
 * Manages allocation, probing, and replacement of clusters using a lockless
 * architecture suitable for SMP search.
 */
class TranspositionTable {
private:
  Cluster *table_ = nullptr; ///< Pointer to the heap-allocated clusters.
  size_t clusterCount_ = 0;  ///< Number of available clusters.
  size_t indexMask_ = 0;     ///< Bitmask for fast modulo addressing.
  uint8_t currentAge_ = 0;   ///< Current generation age (used to age out old entries).
  bool disableTT_ = false;   ///< Flag to disable the TT entirely.
  int numThreads_ = 1;       ///< Number of threads sharing the table.
  bool usingHugePages_ = false; ///< True if the memory was allocated via HugePages.

public:
  TranspositionTable() = default;
  ~TranspositionTable();

  /**
   * @brief Resizes the Transposition Table to the specified capacity in Megabytes.
   */
  void resize(size_t mb);

  /**
   * @brief Clears the contents of the Transposition Table.
   */
  void clear();

  /**
   * @brief Increments the internal age generation to age out older entries.
   */
  void newSearch() { currentAge_ = (currentAge_ + 1) & 0x1F; }

  void setDisabled(bool disabled) { disableTT_ = disabled; }
  void setNumThreads(int n) { numThreads_ = n; }

  /**
   * @brief Calculates the utilization of the Transposition Table (per-mille).
   */
  int hashfull() const;

  /**
   * @brief Issues a memory prefetch instruction for the cluster corresponding to the key.
   */
  void prefetch(uint64_t key) const;

  /**
   * @brief Fully probes the TT for a search entry matching the key.
   * 
   * @return True if a valid entry was found and the output parameters populated.
   */
  bool probe(uint64_t key, int depth, int &score, Move &bestMove,
             int &entryDepth, Bound &entryBound) const;

  /**
   * @brief Lightly probes the TT just to extract the best move (for move ordering).
   */
  Move probeMove(uint64_t key) const;

  /**
   * @brief Lightly probes the TT to extract only the score (for pruning heuristics).
   */
  std::optional<int> probeScore(uint64_t key) const;

  /**
   * @brief Probes the dedicated static evaluation slot.
   */
  bool probeEval(uint64_t key, int16_t &score) const;

  /**
   * @brief Stores a static evaluation in the dedicated slot.
   */
  void storeEval(uint64_t key, int16_t score);

  /**
   * @brief Stores search results into the best slot in the cluster.
   * 
   * Implements a replacement scheme based on depth and age.
   */
  void store(uint64_t key, int depth, int score, Bound bound, Move bestMove,
             int staticEval = 0);

  // Persistence and debugging
  void dumpToFile(const std::string &filename) const;
  bool saveFullToFile(const std::string &filename) const;
  bool loadFullFromFile(const std::string &filename);

private:
  size_t index(uint64_t key) const { return key & indexMask_; }
  int ageDiff(uint8_t entryAge) const {
    return (currentAge_ >= entryAge) ? (currentAge_ - entryAge)
                                     : (32 - entryAge + currentAge_);
  }
};