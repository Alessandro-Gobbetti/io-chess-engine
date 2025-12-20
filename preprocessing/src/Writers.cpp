#include "Writers.hpp"

// --- DatasetWriter Implementation ---

DatasetWriter::DatasetWriter(const std::string& fname, size_t f_size, size_t batch_capacity) 
    : filename(fname), feature_size(f_size), batch_size(batch_capacity) {
    std::ofstream file(filename, std::ios::binary);
    // Pre-allocate memory to avoid reallocations and heap fragmentation
    buffer.reserve(feature_size * batch_size);
}

DatasetWriter::~DatasetWriter() {
    flush();
}

void DatasetWriter::add(const std::array<uint8_t, 42 * 64>& features) {
    buffer.insert(buffer.end(), features.begin(), features.end());
    
    if (buffer.size() >= feature_size * batch_size) {
        flush();
    }
}

void DatasetWriter::add(const std::vector<uint8_t>& features) {
    if (features.size() != feature_size) {
        std::cerr << "Error: Feature size mismatch!" << std::endl;
        return;
    }
    buffer.insert(buffer.end(), features.begin(), features.end());
    if (buffer.size() >= feature_size * batch_size) {
        flush();
    }
}

void DatasetWriter::flush() {
    if (buffer.empty()) return;
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
        buffer.clear();
    }
}

// --- LabelWriter Implementation ---

LabelWriter::LabelWriter(const std::string& fname, size_t batch_capacity) 
    : filename(fname), batch_size(batch_capacity) {
    std::ofstream file(filename, std::ios::binary);
    buffer.reserve(batch_size);
}

LabelWriter::~LabelWriter() {
    flush();
}

void LabelWriter::add(float label) {
    buffer.push_back(label);
    if (buffer.size() >= batch_size) {
        flush();
    }
}

void LabelWriter::flush() {
    if (buffer.empty()) return;
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(float));
        buffer.clear();
    }
}
