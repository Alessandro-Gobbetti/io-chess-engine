#pragma once
// TT.h - Transposition Table for search (3+1 Cluster Architecture)
// 64-byte cache-line aligned clusters with 3 search entries + 1 eval entry

#include "../Types.h"
#include <atomic>
#include <cstdint>

#include <optional>

// --- Constants// Bound enum
enum class Bound : uint8_t {
  NONE = 0,
  UPPER = 1,
  LOWER = 2,
  EXACT = 3,
  LAZY_MASK = 4
};

// --- 1. Search Entry (16 bytes) - Uses XOR checksum for lockless access ---
struct TTEntry {
  // Layout: [Move:16][Score:16][StaticEval:16][Depth:8][Bound:3][Age:5] = 64
  // bits
  uint64_t data;
  // Signature = Key ^ Data (for tearing detection)
  uint64_t signature;

  // Probe: Returns true if signature matches (valid entry)
  bool probe(uint64_t key, uint64_t &out_data) const {
    uint64_t d = __atomic_load_n(&data, __ATOMIC_RELAXED);
    uint64_t s = __atomic_load_n(&signature, __ATOMIC_RELAXED);
    if ((s ^ d) == key) {
      out_data = d;
      return true;
    }
    return false;
  }

  // Save: XORs key with data to sign the entry
  void save(uint64_t key, int16_t score, int16_t sEval, uint8_t depth,
            Bound bound, uint16_t move, uint8_t age) {
    uint64_t newData =
        (static_cast<uint64_t>(move) << 48) |
        (static_cast<uint64_t>(static_cast<uint16_t>(score))
         << 32) | // Implicit cast up
        (static_cast<uint64_t>(static_cast<uint16_t>(sEval)) << 16) |
        (static_cast<uint64_t>(depth) << 8) |
        (static_cast<uint64_t>(static_cast<uint8_t>(bound) & 0x7)
         << 5) |      // 3 bits at pos 5
        (age & 0x1F); // 5 bits at pos 0

    __atomic_store_n(&signature, key ^ newData, __ATOMIC_RELAXED);
    __atomic_store_n(&data, newData, __ATOMIC_RELAXED);
  }

  // Helpers for extracting fields
  static uint16_t getMove(uint64_t d) { return static_cast<uint16_t>(d >> 48); }
  static int16_t getScore(uint64_t d) {
    return static_cast<int16_t>((d >> 32) & 0xFFFF);
  }
  static int16_t getStaticEval(uint64_t d) {
    return static_cast<int16_t>((d >> 16) & 0xFFFF);
  }
  static uint8_t getDepth(uint64_t d) {
    return static_cast<uint8_t>((d >> 8) & 0xFF);
  }
  static Bound getBound(uint64_t d) {
    return static_cast<Bound>((d >> 5) & 0x7);
  }
  static uint8_t getAge(uint64_t d) { return static_cast<uint8_t>(d & 0x1F); }
};

// --- 2. Eval Entry (16 bytes) - Dedicated static eval cache ---
struct alignas(16) EvalEntry {
  // Layout: [Payload:8][Signature:8]
  // Payload: Holds Score (16-bit) and implicit padding
  // Signature: Key ^ Payload (XOR checksum) - enables tearing detection

  std::atomic<uint64_t> payload;
  std::atomic<uint64_t> signature;

  void store(uint64_t key, int16_t score) {
    uint64_t p =
        static_cast<uint64_t>(static_cast<uint16_t>(score)); // Lower 16 bits
    uint64_t s = key ^ p;

    // writing signature first or payload first doesn't matter strictly
    // for correctness with XOR check, but splitting can happen.
    // Relaxed is fine because we check consistency on read.
    signature.store(s, std::memory_order_relaxed);
    payload.store(p, std::memory_order_relaxed);
  }

  bool probe(uint64_t key, int16_t &score) const {
    uint64_t p = payload.load(std::memory_order_relaxed);
    uint64_t s = signature.load(std::memory_order_relaxed);

    // Logic: If (Signature ^ Payload) == Key, then data is consistent (not
    // torn) AND it belongs to this key.
    if ((s ^ p) == key) {
      score = static_cast<int16_t>(p & 0xFFFF);
      return true;
    }
    return false;
  }
};

// --- 3. The 3+1 Cluster (64 bytes = 1 cache line) ---
struct alignas(64) Cluster {
  TTEntry entries[3]; // 48 bytes
  EvalEntry eval;     // 16 bytes (was 8 + 8 padding)
  // No padding needed! 48 + 16 = 64
};
static_assert(sizeof(Cluster) == 64, "Cluster must be 64 bytes");

// --- 4. The Main Table Class ---
class TranspositionTable {
private:
  Cluster *table_ = nullptr;
  size_t clusterCount_ = 0;
  size_t indexMask_ = 0;
  uint8_t currentAge_ = 0;
  bool disableTT_ = false;
  bool debugMode_ = false;
  int numThreads_ = 1;
  bool usingHugePages_ = false;

public:
  TranspositionTable() = default;
  ~TranspositionTable();

  void resize(size_t mb);
  void clear();
  void newSearch() {
    currentAge_ = (currentAge_ + 1) & 0x1F;
  } // 5-bit age wraps at 32
  void setDisabled(bool disabled) { disableTT_ = disabled; }
  void setNumThreads(int n) { numThreads_ = n; }
  int hashfull() const;
  void prefetch(uint64_t key) const;

  // --- Search Probing (checks all 3 slots in cluster) ---
  bool probe(uint64_t key, int depth, int alpha, int beta, int &score,
             Move &bestMove, int &entryDepth, Bound &entryBound) const;

  // Lightweight Probe - only extracts TT move for move ordering
  Move probeMove(uint64_t key) const;

  // Lightweight Probe - only extracts TT score for pruning decisions
  std::optional<int> probeScore(uint64_t key) const;

  // --- Eval Probing (dedicated slot) ---
  bool probeEval(uint64_t key, int16_t &score) const;
  void storeEval(uint64_t key, int16_t score);

  // --- Smart Replacement Store ---
  void store(uint64_t key, int depth, int score, Bound bound, Move bestMove,
             int staticEval = 0);

  // Debug
  void dumpToFile(const std::string &filename) const;
  void setDebugMode(bool enabled) { debugMode_ = enabled; }
  bool isDebugMode() const { return debugMode_; }

private:
  size_t index(uint64_t key) const { return key & indexMask_; }
  int ageDiff(uint8_t entryAge) const {
    return (currentAge_ >= entryAge) ? (currentAge_ - entryAge)
                                     : (32 - entryAge + currentAge_);
  }
};