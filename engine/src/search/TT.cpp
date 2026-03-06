#include "TT.h"
#include "../Types.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#ifdef __linux__
#include <sys/mman.h>
#endif

TranspositionTable::~TranspositionTable() {
  if (table_) {
#ifdef __linux__
    if (usingHugePages_) {
      munmap(table_, clusterCount_ * sizeof(Cluster));
    } else {
      free(table_);
    }
#else
    free(table_);
#endif
  }
}

void TranspositionTable::resize(size_t mb) {
  size_t bytes = mb * 1024 * 1024;
  size_t requested = bytes / sizeof(Cluster);

  // Power-of-two sizing for fast masking
  clusterCount_ = 1;
  while ((clusterCount_ << 1) <= requested)
    clusterCount_ <<= 1;
  indexMask_ = clusterCount_ - 1;

  size_t oldClusterCount = clusterCount_;
  // Free old table
  if (table_) {
#ifdef __linux__
    if (usingHugePages_) {
      munmap(table_, oldClusterCount * sizeof(Cluster));
    } else {
      free(table_);
    }
#else
    free(table_);
#endif
    table_ = nullptr;
  }

  size_t allocSize = clusterCount_ * sizeof(Cluster);

#ifdef __linux__
  // Try Linux Huge Pages first (requires: sudo sysctl -w vm.nr_hugepages=N)
  table_ = static_cast<Cluster *>(
      mmap(nullptr, allocSize, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0));

  if (table_ != MAP_FAILED) {
    usingHugePages_ = true;
    std::cout << "info string TT: Using Linux Huge Pages (" << mb << " MB, "
              << clusterCount_ << " clusters)" << std::endl;
  } else {
    usingHugePages_ = false;
    // Fallback to aligned_alloc
    table_ = static_cast<Cluster *>(aligned_alloc(64, allocSize));
    if (!table_) {
      std::cerr << "FATAL: TT allocation failed!" << std::endl;
      return;
    }
    std::cout << "info string TT: Using aligned_alloc (" << mb << " MB)"
              << std::endl;
  }
#else
  // macOS/Windows: Use aligned_alloc or posix_memalign
  usingHugePages_ = false;
  if (posix_memalign(reinterpret_cast<void **>(&table_), 64, allocSize) != 0) {
    std::cerr << "FATAL: TT allocation failed!" << std::endl;
    table_ = nullptr;
    return;
  }
  std::cout << "info string TT: 3+1 Cluster (" << mb << " MB, " << clusterCount_
            << " clusters)" << std::endl;
#endif

  clear();
}

void TranspositionTable::clear() {
  if (!table_)
    return;
  std::memset(table_, 0, clusterCount_ * sizeof(Cluster));
  currentAge_ = 0;
}

void TranspositionTable::prefetch(uint64_t key) const {
  if (!table_)
    return;
  __builtin_prefetch(&table_[index(key)]);
}

bool TranspositionTable::probe(uint64_t key, int depth, int alpha, int beta,
                               int &score, Move &bestMove, int &entryDepth,
                               Bound &entryBound) const {
  if (disableTT_ || !table_)
    return false;

  const Cluster &cluster = table_[index(key)];
  int maxMoveDepth = -1;
  bool cutoffFound = false;

  // Check all 3 slots in the cluster
  for (int i = 0; i < 3; ++i) {
    uint64_t d;
    if (cluster.entries[i].probe(key, d)) {
      // Found matching entry
      uint16_t move = TTEntry::getMove(d);
      uint8_t eDepth = TTEntry::getDepth(d);
      uint8_t eAge = TTEntry::getAge(d);
      Bound boundRaw = TTEntry::getBound(d);

      // Strip Lazy Flag for logic
      Bound bound = static_cast<Bound>(static_cast<uint8_t>(boundRaw) & 0x3);
      int16_t eScore = TTEntry::getScore(d);

      // Export entry data regardless of depth/age checks
      // This is crucial for heuristics like Singular Extensions
      // which analyze TT entries that don't meet cutoff requirements
      score = eScore;
      entryDepth = static_cast<int>(eDepth);
      entryBound = boundRaw;

      // Improved Move Ordering Logic
      if (move != 0) {
        int priority = static_cast<int>(eDepth) - ageDiff(eAge);
        if (priority > maxMoveDepth) {
          maxMoveDepth = priority;
          bestMove = Move(move);
        }
      }

      // Depth check for score cutoff
      if (static_cast<int>(eDepth) < depth)
        continue;

      // Age check - only trust scores from current search for CUTOFFS
      if (eAge != currentAge_)
        continue;

      if (bound == Bound::EXACT)
        return true;
      if (bound == Bound::UPPER && score <= alpha)
        return true;
      if (bound == Bound::LOWER && score >= beta)
        return true;
    }
  }

  return false;
}

Move TranspositionTable::probeMove(uint64_t key) const {
  if (disableTT_ || !table_)
    return Move();

  const Cluster &cluster = table_[index(key)];
  Move bestFound = Move();
  int maxDepth = -1;

  for (int i = 0; i < 3; ++i) {
    uint64_t d;
    if (cluster.entries[i].probe(key, d)) {
      // Relaxed age check: accept moves from any generation
      uint16_t move = TTEntry::getMove(d);
      if (move != 0) {
        int eDepth = static_cast<int>(TTEntry::getDepth(d));
        uint8_t eAge = TTEntry::getAge(d);
        int priority = eDepth - ageDiff(eAge);

        if (priority > maxDepth) {
          maxDepth = priority;
          bestFound = Move(move);
        }
      }
    }
  }

  return bestFound;
}

std::optional<int> TranspositionTable::probeScore(uint64_t key) const {
  if (disableTT_ || !table_)
    return std::nullopt;

  const Cluster &cluster = table_[index(key)];

  for (int i = 0; i < 3; ++i) {
    uint64_t d;
    if (cluster.entries[i].probe(key, d)) {
      uint8_t eAge = TTEntry::getAge(d);
      if (eAge == currentAge_) {
        return static_cast<int>(TTEntry::getScore(d));
      }
    }
  }

  return std::nullopt;
}

// --- Dedicated Eval Cache ---
bool TranspositionTable::probeEval(uint64_t key, int16_t &score) const {
  if (!table_)
    return false;

  const Cluster &cluster = table_[index(key)];

  // 1. Check dedicated eval entry (fastest)
  if (cluster.eval.probe(key, score))
    return true;

  // 2. Fallback: Check search entries!
  // They contain staticEval, which might be valid even if search score isn't.
  for (int i = 0; i < 3; ++i) {
    uint64_t d;
    if (cluster.entries[i].probe(key, d)) {
      // Found signature match -> return the stored static eval
      Bound bound = TTEntry::getBound(d);

      // SAFETY: If this was a Lazy Skip entry, the score stored is 'fastScore'
      // (Material), not a true NN evaluation. We MUST NOT use it as a static
      // eval proxy for standard pruning judgments.
      if (static_cast<uint8_t>(bound) & static_cast<uint8_t>(Bound::LAZY_MASK))
        continue;

      score = TTEntry::getStaticEval(d);

      // Filter out invalid static evals
      if (score == SearchConstants::INVALID_EVAL)
        continue;

      return true;
    }
  }

  return false;
}

void TranspositionTable::storeEval(uint64_t key, int16_t score) {
  if (!table_)
    return;
  table_[index(key)].eval.store(key, score);
}

// --- Smart Replacement Store ---
void TranspositionTable::store(uint64_t key, int depth, int score, Bound bound,
                               Move bestMove, int staticEval) {
  if (!table_)
    return;

  Cluster &cluster = table_[index(key)];
  TTEntry *replaceTarget = &cluster.entries[0];
  int minPriority = 100000;

  for (int i = 0; i < 3; ++i) {
    uint64_t d;
    if (cluster.entries[i].probe(key, d)) {
      // Found exact key match - always update this slot
      replaceTarget = &cluster.entries[i];

      // Preserve move if new entry has no move
      if (bestMove.move() == 0) {
        uint16_t oldMove = TTEntry::getMove(d);
        if (oldMove != 0)
          bestMove = Move(oldMove);
      }
      break;
    }

    // Calculate replaceability score: prefer replacing shallow/stale entries
    d = __atomic_load_n(&cluster.entries[i].data, __ATOMIC_RELAXED);
    uint8_t entryAge = TTEntry::getAge(d);
    uint8_t entryDepth = TTEntry::getDepth(d);

    int ageDelta = ageDiff(entryAge);
    int priority =
        static_cast<int>(entryDepth) - (ageDelta * 4); // Stale = lower priority

    // Bonus for EXACT entries (more valuable)
    Bound storedBound = TTEntry::getBound(d);
    if (storedBound == Bound::EXACT)
      priority += 4;

    // Penalty for LAZY entries (less valuable than real search)
    if (static_cast<uint8_t>(storedBound) &
        static_cast<uint8_t>(Bound::LAZY_MASK))
      priority -= 10;

    // Bonus for having a move (valuable for ordering)
    if (TTEntry::getMove(d) != 0)
      priority += 2;

    if (priority < minPriority) {
      minPriority = priority;
      replaceTarget = &cluster.entries[i];
    }
  }

  // Store the entry
  // Clamp score and staticEval to int16_t range to prevent overflow
  // but preserve INVALID_EVAL (32001) since it acts as a sentinel.
  int16_t clampedScore = static_cast<int16_t>(score == SearchConstants::INVALID_EVAL ? score : std::clamp(score, -32000, 32000));
  int16_t clampedEval = static_cast<int16_t>(staticEval == SearchConstants::INVALID_EVAL ? staticEval : std::clamp(staticEval, -32000, 32000));
  replaceTarget->save(key, clampedScore,
                      clampedEval,
                      static_cast<uint8_t>(std::max(0, depth)), bound,
                      static_cast<uint16_t>(bestMove.move()), currentAge_);
}

int TranspositionTable::hashfull() const {
  if (clusterCount_ == 0 || !table_)
    return 0;

  int used = 0;
  size_t samplesToCheck = std::min<size_t>(1000, clusterCount_);

  for (size_t i = 0; i < samplesToCheck; ++i) {
    const Cluster &cluster = table_[i];
    for (int j = 0; j < 3; ++j) {
      uint64_t d = __atomic_load_n(&cluster.entries[j].data, __ATOMIC_RELAXED);
      if (d != 0) { // Count any occupied slot
        used++;
        break; // Count each cluster only once
      }
    }
  }

  return (used * 1000) / static_cast<int>(samplesToCheck);
}

void TranspositionTable::dumpToFile(const std::string &filename) const {
  if (!table_)
    return;

  std::ofstream out(filename, std::ios::binary);
  if (!out)
    return;

  // Header
  out.write(reinterpret_cast<const char *>(&clusterCount_),
            sizeof(clusterCount_));
  out.write(reinterpret_cast<const char *>(&currentAge_), sizeof(currentAge_));

  // Write non-empty entries
  for (size_t i = 0; i < clusterCount_; ++i) {
    const Cluster &cluster = table_[i];
    for (int j = 0; j < 3; ++j) {
      uint64_t d = __atomic_load_n(&cluster.entries[j].data, __ATOMIC_RELAXED);
      uint64_t s =
          __atomic_load_n(&cluster.entries[j].signature, __ATOMIC_RELAXED);
      if (d != 0) {
        out.write(reinterpret_cast<const char *>(&i), sizeof(i));
        out.write(reinterpret_cast<const char *>(&j), sizeof(j));
        out.write(reinterpret_cast<const char *>(&d), sizeof(d));
        out.write(reinterpret_cast<const char *>(&s), sizeof(s));
      }
    }
  }
}