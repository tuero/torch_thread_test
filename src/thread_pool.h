// File: thread_pool.h
// Description: Simple thread pool class to dispatch threads continuously on input

#ifndef THREAD_POOL_H_
#define THREAD_POOL_H_

#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "queue.h"


// Create a thread pool object.
template <typename InputT, typename OutputT>
class ThreadPool {
public:
    ThreadPool() = delete;

    /**
     * Create a thread pool object.
     * @param num_threads Number of threads the pool should run
     */
    ThreadPool(int num_threads)
        : num_threads(num_threads),
          queue_input_(std::numeric_limits<uint16_t>::max()),
          queue_output_(std::numeric_limits<uint16_t>::max()) {}

    /**
     * Run the given function on the thread pool.
     * @param func Function to run in parallel, which should match templated arguments for input and output
     * @param inputs Input items for each job, gets passed to the given function
     * @return Vector of results, in order of given jobs during construction
     */
    std::vector<OutputT> run(std::function<OutputT(InputT)> func, const std::vector<InputT>& inputs) {
        // Populate queue
        int id = -1;
        for (auto const& job : inputs) {
            queue_input_.Push({job, ++id});
        }

        // Start N threads
        threads_.reserve(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this, func]() { this->thread_runner(func); });
        }

        // Wait for all to complete
        for (auto& t : threads_) {
            t.join();
        }
        threads_.clear();

        // Compile results, such that the id is in order to match passed order
        std::vector<OutputT> results;
        results.reserve(queue_output_.Size());
        std::map<int, OutputT> result_map;
        while (!queue_output_.Empty()) {
            std::optional<QueueItemOutput> result = queue_output_.Pop();
            result_map.emplace(result->id, result->output);
        }
        for (auto const& result : result_map) {
            results.push_back(result.second);
        }

        return results;
    }

private:
    struct QueueItemInput {    // Wrapper for input type with id
        InputT input;
        int id;
    };

    struct QueueItemOutput {    // Wrapper for output type with id
        OutputT output;
        int id;
    };

    // Runner for each thread, runs given function and pulls next item from input jobs if available
    void thread_runner(std::function<OutputT(InputT)> func) {
        while (true) {
            std::optional<QueueItemInput> item;
            {
                std::unique_lock<std::mutex> lock(queue_input_m_);
                // Jobs are done, thread can stop
                if (queue_input_.Empty()) {
                    break;
                }
                item = queue_input_.Pop();
            }

            // Run job
            OutputT result = func(item->input);

            // Store result
            {
                std::unique_lock<std::mutex> lock(queue_output_m_);
                queue_output_.Push({result, item->id});
            }
        }
    }

    int num_threads;                                 // How many threads in the pool
    std::vector<std::thread> threads_;               // Threads in the pool
    ThreadedQueue<QueueItemInput> queue_input_;      // Queue for input argument for job
    ThreadedQueue<QueueItemOutput> queue_output_;    // Queue for output return values for job
    std::mutex queue_input_m_;                       // Mutex for the input queue
    std::mutex queue_output_m_;                      // Musted for the output queue
};

#endif    // THREAD_POOL_H_