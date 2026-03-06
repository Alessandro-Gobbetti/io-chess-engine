#include "Writers.hpp"

// --- DatasetWriter Implementation ---

DatasetWriter::DatasetWriter(const std::string &fname, size_t batch_capacity)
    : filename(fname), batch_size(batch_capacity) {

  // Open in binary mode, truncate existing file
  std::ofstream file(filename, std::ios::binary);

  // Reserve memory to prevent reallocations
  buffer.reserve(batch_size);
}

DatasetWriter::~DatasetWriter() { flush(); }

void DatasetWriter::add(const ChessInput &features) {
  buffer.push_back(features);

  if (buffer.size() >= batch_size) {
    flush();
  }
}

void DatasetWriter::flush() {
  if (buffer.empty())
    return;

  std::ofstream file(filename, std::ios::binary | std::ios::app);
  if (file.is_open()) {
    // Write the raw memory of the vector directly
    // size in bytes = number of elements * size of one struct
    file.write(reinterpret_cast<const char *>(buffer.data()),
               buffer.size() * sizeof(ChessInput));
    buffer.clear();
  }
}

// --- LabelWriter Implementation ---

LabelWriter::LabelWriter(const std::string &fname, size_t batch_capacity)
    : filename(fname), batch_size(batch_capacity) {
  std::ofstream file(filename, std::ios::binary);
  buffer.reserve(batch_size);
}

LabelWriter::~LabelWriter() { flush(); }

void LabelWriter::add(float label) {
  buffer.push_back(label);
  if (buffer.size() >= batch_size) {
    flush();
  }
}

void LabelWriter::flush() {
  if (buffer.empty())
    return;
  std::ofstream file(filename, std::ios::binary | std::ios::app);
  if (file.is_open()) {
    file.write(reinterpret_cast<const char *>(buffer.data()),
               buffer.size() * sizeof(float));
    buffer.clear();
  }
}

// --- WDLWriter Implementation ---

WDLWriter::WDLWriter(const std::string &fname, size_t batch_capacity)
    : filename(fname), batch_size(batch_capacity * 4) { // 4 floats per sample
  // Create/truncate file
  std::ofstream file(filename, std::ios::binary);
  buffer.reserve(batch_size);
}

WDLWriter::~WDLWriter() { flush(); }

void WDLWriter::add(const WDLOutput &wdl) {
  buffer.push_back(wdl.win);
  buffer.push_back(wdl.draw);
  buffer.push_back(wdl.loss);
  buffer.push_back(wdl.mate_dist);

  if (buffer.size() >= batch_size) {
    flush();
  }
}

void WDLWriter::flush() {
  if (buffer.empty())
    return;
  std::ofstream file(filename, std::ios::binary | std::ios::app);
  if (file.is_open()) {
    file.write(reinterpret_cast<const char *>(buffer.data()),
               buffer.size() * sizeof(float));
    buffer.clear();
  }
}

// --- ExpertWeightsWriter Implementation ---

ExpertWeightsWriter::ExpertWeightsWriter(const std::string &fname,
                                         size_t batch_capacity)
    : filename(fname), batch_size(batch_capacity * NUM_EXPERTS) {
  // Create/truncate file
  std::ofstream file(filename, std::ios::binary);
  // Reserve buffer for batch_capacity positions * NUM_EXPERTS weights per
  // position
  buffer.reserve(batch_size);
}

ExpertWeightsWriter::~ExpertWeightsWriter() { flush(); }

void ExpertWeightsWriter::add(const ExpertRouter::ExpertWeights &weights) {
  add(weights.weights);
}

void ExpertWeightsWriter::add(const float *weights) {
  // Append all 6 weights to buffer
  for (int i = 0; i < NUM_EXPERTS; ++i) {
    buffer.push_back(weights[i]);
  }

  if (buffer.size() >= batch_size) {
    flush();
  }
}

void ExpertWeightsWriter::flush() {
  if (buffer.empty())
    return;
  std::ofstream file(filename, std::ios::binary | std::ios::app);
  if (file.is_open()) {
    file.write(reinterpret_cast<const char *>(buffer.data()),
               buffer.size() * sizeof(float));
    buffer.clear();
  }
}

// --- ExpertDatasetWriter Implementation ---

static const char *EXPERT_NAMES[] = {"Tactical",      "Strategic",
                                     "Major Endgame", "Minor Endgame",
                                     "Survivor",      "Killer"};

ExpertDatasetWriter::ExpertDatasetWriter(const std::string &output_dir,
                                         size_t batch_capacity) {
  counts.fill(0);

  for (int i = 0; i < NUM_EXPERTS; ++i) {
    std::string feat_path =
        output_dir + "/expert" + std::to_string(i) + "_features.bin";
    std::string label_path =
        output_dir + "/expert" + std::to_string(i) + "_labels.bin";

    feature_writers[i] =
        std::make_unique<DatasetWriter>(feat_path, batch_capacity);
    label_writers[i] = std::make_unique<WDLWriter>(label_path, batch_capacity);
  }
}

ExpertDatasetWriter::~ExpertDatasetWriter() { flush(); }

void ExpertDatasetWriter::add(const ChessInput &features, const WDLOutput &wdl,
                              const ExpertRouter::ExpertWeights &weights) {
  // Each expert has INDEPENDENT inclusion criteria based on raw scores
  // (before softmax normalization). No comparisons between experts.

  // NEW: Threshold lowered to include BOTH primary AND secondary experts
  // SCORE_ACTIVE = 3.0, SCORE_SECONDARY = 2.0
  // Using 1.8 means both PRIMARY (3.0) and SECONDARY (2.0) experts qualify
  // This ensures overlap for edge positions and meaningful top-2 training
  static constexpr float BASE_EXPERT_THRESHOLD = 1.8f;

  // SAFETY: Always include the PRIMARY expert (argmax) to ensure no position is
  // dropped
  int best_expert = 0;
  float best_score = weights.raw_scores[0];

  for (int i = 1; i < ExpertRouter::NUM_BASE; ++i) {
    if (weights.raw_scores[i] > best_score) {
      best_score = weights.raw_scores[i];
      best_expert = i;
    }
  }

  // Add to base experts
  for (int i = 0; i < ExpertRouter::NUM_BASE; ++i) {
    // Condition: Be the best expert OR exceed the threshold
    if (i == best_expert || weights.raw_scores[i] > BASE_EXPERT_THRESHOLD) {
      feature_writers[i]->add(features);
      label_writers[i]->add(wdl);
      counts[i]++;
    }
  }

  // Aux expert 4: Survivor - when losing (gate > threshold)
  if (weights.weights[4] > AUX_GATE_THRESHOLD) {
    feature_writers[4]->add(features);
    label_writers[4]->add(wdl);
    counts[4]++;
  }

  // Aux expert 5: Killer - when winning (gate > threshold)
  if (weights.weights[5] > AUX_GATE_THRESHOLD) {
    feature_writers[5]->add(features);
    label_writers[5]->add(wdl);
    counts[5]++;
  }
}

void ExpertDatasetWriter::flush() {
  for (int i = 0; i < NUM_EXPERTS; ++i) {
    if (feature_writers[i])
      feature_writers[i]->flush();
    if (label_writers[i])
      label_writers[i]->flush();
  }
}

void ExpertDatasetWriter::print_stats() const {
  std::cout << "\nExpert Dataset Statistics:" << std::endl;
  size_t total = 0;
  for (int i = 0; i < NUM_EXPERTS; ++i) {
    std::cout << "  Expert " << i << " (" << EXPERT_NAMES[i]
              << "): " << counts[i] << " samples" << std::endl;
    total += counts[i];
  }
  // Note: total may be > processed because aux experts can overlap with base
  std::cout << "  Total entries: " << total << std::endl;
}
