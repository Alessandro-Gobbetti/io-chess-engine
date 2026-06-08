#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <algorithm>

/**
 * @class PersistentThreadPool
 * @brief A generic thread pool used for intra-evaluation parallelism and batch processing.
 *
 * This thread pool is used by the `MoEDoubleAccumulator` in the core engine 
 * when the UCI `EvalThreads` option is set to >1. It parallelizes the evaluation 
 * of the neural network's internal spatial branches.
 * 
 * It is also used heavily by the standalone benchmark tools in `src/tools/` 
 * to evaluate massive batches of positions simultaneously.
 */
class PersistentThreadPool {
public:
  explicit PersistentThreadPool(int workers) {
    const int n = std::max(0, workers);
    for (int i = 0; i < n; ++i)
      threads_.emplace_back(&PersistentThreadPool::worker_loop, this);
  }

  ~PersistentThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &t : threads_) {
      if (t.joinable())
        t.join();
    }
  }

  PersistentThreadPool(const PersistentThreadPool &) = delete;
  PersistentThreadPool &operator=(const PersistentThreadPool &) = delete;

  void parallel_for(int n, const std::function<void(int)> &fn) {
    if (n <= 0)
      return;
    if (threads_.empty() || n == 1) {
      for (int i = 0; i < n; ++i)
        fn(i);
      return;
    }

    auto job = std::make_shared<Job>();
    job->total = n;
    job->next.store(0, std::memory_order_relaxed);
    job->remaining.store(n, std::memory_order_relaxed);
    job->fn = fn;

    {
      std::lock_guard<std::mutex> lk(mu_);
      for (size_t i = 0; i < threads_.size(); ++i)
        jobs_.push(job);
    }
    cv_.notify_all();

    run_job(job);

    std::unique_lock<std::mutex> lk(job->doneMu);
    job->doneCv.wait(lk, [&] {
      return job->remaining.load(std::memory_order_acquire) == 0;
    });
  }

private:
  struct Job {
    int total = 0;
    std::atomic<int> next{0};
    std::atomic<int> remaining{0};
    std::function<void(int)> fn;
    std::mutex doneMu;
    std::condition_variable doneCv;
  };

  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::queue<std::shared_ptr<Job>> jobs_;
  bool stop_ = false;

  static void finish_one(const std::shared_ptr<Job> &job) {
    if (job->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard<std::mutex> lk(job->doneMu);
      job->doneCv.notify_one();
    }
  }

  static void run_job(const std::shared_ptr<Job> &job) {
    while (true) {
      const int idx = job->next.fetch_add(1, std::memory_order_relaxed);
      if (idx >= job->total)
        break;
      job->fn(idx);
      finish_one(job);
    }
  }

  void worker_loop() {
    while (true) {
      std::shared_ptr<Job> job;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || !jobs_.empty(); });
        if (stop_ && jobs_.empty())
          return;
        job = jobs_.front();
        jobs_.pop();
      }
      run_job(job);
    }
  }
};
