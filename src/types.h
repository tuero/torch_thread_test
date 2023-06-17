// File: types.h
// Description: Basic types used throughout codebase

#ifndef TYPES_H_
#define TYPES_H_

#include <atomic>
#include <vector>

// Observation struct for parameterizing model input
using Observation = std::vector<float>;
struct ObservationShape {
    int c;
    int h;
    int w;
};

struct InferenceOutput {
    std::vector<double> logits;
    std::vector<double> policy;
    std::vector<double> log_policy;
    double heuristic;
};

class StopToken {
public:
    StopToken() : flag_(false) {}
    void stop() {
        flag_ = true;
    }
    bool stop_requested() const {
        return flag_;
    }

private:
    std::atomic<bool> flag_;
};

#endif    // TYPES_H_
