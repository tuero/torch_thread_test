#ifndef HREAD_SAFE_QUEUE_H_
#define HREAD_SAFE_QUEUE_H_

#include <iostream>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

// A threadsafe-queue.
template <class T>
class ThreadedQueue {
public:
    explicit ThreadedQueue(int max_size) : max_size_(max_size) {}

    bool Push(const T& value) {
        std::unique_lock<std::mutex> lock(m_);
        while ((int)q_.size() >= max_size_) {
            cv_.wait(lock);
        }
        q_.push(value);
        cv_.notify_one();
        return true;
    }

    std::optional<T> Pop() {
        std::unique_lock<std::mutex> lock(m_);
        while (q_.empty()) {
            if (block_new_values_) {
                return std::nullopt;
            }
            cv_.wait(lock);
        }
        T val = q_.front();
        q_.pop();
        cv_.notify_one();
        return val;
    }

    bool Empty() {
        std::unique_lock<std::mutex> lock(m_);
        return q_.empty();
    }

    void Clear() {
        std::unique_lock<std::mutex> lock(m_);
        while (!q_.empty()) {
            q_.pop();
        }
    }

    int Size() {
        std::unique_lock<std::mutex> lock(m_);
        return q_.size();
    }

    // Causes pushing new values to fail. Useful for shutting down the queue.
    void BlockNewValues() {
        std::unique_lock<std::mutex> lock(m_);
        block_new_values_ = true;
        cv_.notify_all();
    }

private:
    bool block_new_values_ = false;
    int max_size_;
    std::queue<T> q_;
    std::mutex m_;
    std::condition_variable cv_;
};



#endif    // HREAD_SAFE_QUEUE_H_