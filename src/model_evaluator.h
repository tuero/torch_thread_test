#ifndef MODEL_EVALUATOR_H_
#define MODEL_EVALUATOR_H_

#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "model.h"
#include "queue.h"
#include "types.h"

// Handles threaded queries for the model
class ModelEvaluator {
public:
    explicit ModelEvaluator(const ObservationShape observation_shape, int num_actions, int search_threads)
        : model_wrapper(observation_shape, num_actions), queue_(search_threads * 4) {
        inference_threads_.emplace_back([this]() { this->InferenceRunner(); });
    };

    ~ModelEvaluator() {
        // Clear the incoming queues and stop oustanding threads
        stop_token_.stop();
        queue_.BlockNewValues();
        queue_.Clear();
        for (auto& t : inference_threads_) {
            t.join();
        }
    }

    /**
     * Perform inference for a group of observations by sending to thread runner
     */
    std::vector<InferenceOutput> Inference(std::vector<Observation>& inference_inputs) {
        std::promise<std::vector<InferenceOutput>> prom;
        std::future<std::vector<InferenceOutput>> fut = prom.get_future();
        queue_.Push(QueueItem{inference_inputs, &prom});
        return fut.get();
    }

    void print() const {
        model_wrapper.print();
    }

private:
    // Runner to perform inference queries if using threading on the model
    void InferenceRunner() {
        while (!stop_token_.stop_requested()) {
            std::optional<QueueItem> item = queue_.Pop();
            if (!item) {
                continue;
            }
            std::vector<InferenceOutput> outputs = model_wrapper.Inference(item->inputs);
            item->prom->set_value(outputs);
        }
    }

    TwoHeadedConvNetWrapper model_wrapper;

    // Struct for holding promised value for inference queries
    struct QueueItem {
        std::vector<Observation> inputs;
        std::promise<std::vector<InferenceOutput>>* prom;
    };

    StopToken stop_token_;
    ThreadedQueue<QueueItem> queue_;                // Queue for inference requests
    std::vector<std::thread> inference_threads_;    // Threads for inference requests
};


#endif    // MODEL_EVALUATOR_H_
