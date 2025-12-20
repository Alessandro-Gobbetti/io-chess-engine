#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <array>
#include <cstdint>

class DatasetWriter {
private:
    std::string filename;
    std::vector<uint8_t> buffer;
    size_t feature_size;
    size_t batch_size;

public:
    DatasetWriter(const std::string& fname, size_t f_size, size_t batch_capacity = 1000);
    ~DatasetWriter();
    void add(const std::array<uint8_t, 42 * 64>& features);
    void add(const std::vector<uint8_t>& features);
    void flush();
};

class LabelWriter {
private:
    std::string filename;
    std::vector<float> buffer;
    size_t batch_size;

public:
    LabelWriter(const std::string& fname, size_t batch_capacity = 1000);
    ~LabelWriter();
    void add(float label);
    void flush();
};