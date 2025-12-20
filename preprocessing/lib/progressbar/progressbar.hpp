#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <utility> // for std::declval
#include <chrono>
#include <iomanip>
#include <sstream>
#include <mutex>

// --- Helper: ProgressBar Class ---
class ProgressBar {
private:
    int total;
    int width;
    bool leave;
    int current = 0;
    double precision = 1e-3;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::mutex mutex_;

    std::string format_time(double seconds) {
        int s = static_cast<int>(seconds);
        int m = s / 60;
        s %= 60;
        int h = m / 60;
        m %= 60;
        
        std::stringstream ss;
        if (h > 0) ss << h << ":";
        ss << std::setfill('0') << std::setw(2) << m << ":" << std::setw(2) << s;
        return ss.str();
    }

    void update_internal(int current) {
        // ubdate only if significant change
        if (std::abs(current - this->current) < precision * total && current != total) {
            return;
        }

        this->current = current;
        
        auto now = std::chrono::steady_clock::now();


        double progress = (double)current / total;
        if (progress > 1.0) progress = 1.0;

        double fill_width = width * progress;
        int full_blocks = std::floor(fill_width);
        
        // partials characters for smooth progress bar
        static const std::string partials[] = {"", "▏", "▎", "▍", "▌", "▋", "▊", "▉"};
        
        int partial_idx = (int)((fill_width - full_blocks) * 8);
        // Clamp index to be safe
        if (partial_idx < 0) partial_idx = 0;
        if (partial_idx > 7) partial_idx = 7;

        // Time calculations
        // auto now = std::chrono::steady_clock::now(); // Already captured above
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double rate = (current > 0) ? current / elapsed : 0.0;
        double remaining = (rate > 0) ? (total - current) / rate : 0.0;

        std::cout << "\r" << std::setw(3) << int(progress * 100.0) << "%|"; 
        
        for (int i = 0; i < full_blocks; ++i) {
            std::cout << "█"; 
        }
        
        if (progress < 1.0) {
            std::cout << partials[partial_idx];
        }
        
        int printed_len = full_blocks + (progress < 1.0 && partial_idx > 0 ? 1 : 0);
        for (int i = printed_len; i < width; ++i) {
            std::cout << " "; 
        }
        
        std::cout << "| " << current << "/" << total;
        std::cout << " [" << format_time(elapsed) << "<" << format_time(remaining) 
                  << ", " << std::fixed << std::setprecision(2) << rate << "it/s]";
        
        std::cout << std::flush;
    }

public:
    ProgressBar(int total_, int width_ = 50, bool leave_ = false, double precision_ = 1e-3) 
        : total(total_), width(width_), leave(leave_) {
        start_time = std::chrono::steady_clock::now();
    }

    void update(int current) {
        std::lock_guard<std::mutex> lock(mutex_);
        update_internal(current);
    }

    void complete() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (leave) {
            update_internal(total);
            std::cout << std::endl;
        } else {
            // Clear the line: move to start, overwrite with spaces, move back
            std::cout << "\r" << std::string(width + 60, ' ') << "\r" << std::flush;
        }
    }

    ~ProgressBar() {
        // std::cout << "Done." << std::endl;
        complete();
    }
};

// --- Progress Wrapper ---
template<typename Container>
class ProgressWrapper {
    Container& container;
    ProgressBar pbar;

public:
    ProgressWrapper(Container& c, int width = 50, bool leave = false)
        : container(c), pbar(c.size(), width, leave) {
        pbar.update(0);
    }

    // Deduce iterator type
    using IteratorType = decltype(std::begin(std::declval<Container&>()));

    class Iterator {
        IteratorType it;
        ProgressBar* pbar;
        int idx;

    public:
        Iterator(IteratorType iter, ProgressBar* pb, int i) : it(iter), pbar(pb), idx(i) {}

        Iterator& operator++() {
            ++it;
            ++idx;
            if (pbar) pbar->update(idx);
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return it != other.it;
        }

        auto& operator*() { return *it; }
    };

    Iterator begin() {
        return Iterator(std::begin(container), &pbar, 0);
    }

    Iterator end() {
        return Iterator(std::end(container), &pbar, (int)container.size());
    }
};

template<typename Container>
auto show_progress(Container& c, int width = 50, bool leave = false) {
    return ProgressWrapper<Container>(c, width, leave);
}
