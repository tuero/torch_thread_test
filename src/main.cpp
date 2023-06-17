#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "rnd/stonesngems_base.h"
#include "search.h"
#include "model_evaluator.h"
#include "thread_pool.h"
#include "types.h"

const int NUM_THREADS = 8;
const int ENV_WIDTH = 16;
const int ENV_HEIGHT = 16;
const int ENV_CHANNELS = 36;
const int NUM_ACTIONS = 5;

const ObservationShape OBSERVATION_SHAPE = {ENV_CHANNELS, ENV_HEIGHT, ENV_WIDTH};


int main() {
    // Set torch seed
    torch::manual_seed(0);
    torch::cuda::manual_seed_all(0);
    torch::globalContext().setDeterministicCuDNN(true);
    torch::globalContext().setBenchmarkCuDNN(false);

    ThreadPool<SearchInput, bool> pool(NUM_THREADS);
    std::unique_ptr<ModelEvaluator> evaluator_A = std::make_unique<ModelEvaluator>(OBSERVATION_SHAPE, NUM_ACTIONS, NUM_THREADS);
    std::unique_ptr<ModelEvaluator> evaluator_B = std::make_unique<ModelEvaluator>(OBSERVATION_SHAPE, NUM_ACTIONS, NUM_THREADS);

    std::vector<std::string> board_str {
        "16|16|9999|1|02|02|02|01|01|02|02|02|02|39|02|02|02|02|02|02|02|02|02|02|02|02|02|01|02|02|02|02|02|02|02|02|02|03|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|01|02|02|01|02|02|02|02|02|02|02|02|03|02|02|02|02|02|02|02|01|02|02|02|02|02|39|02|02|02|02|07|01|02|01|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|00|02|02|02|02|02|03|02|02|02|02|02|02|01|02|02|02|02|02|02|01|02|02|02|03|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|01|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|01|02|02|02|02|02|01|02|02|03|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|39|02|02|02|02|02|39|02|02|02|02|02|02|01|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|02|39|02|02|02|02|01|02|02|02|02|02",
    };
    

    std::vector<SearchInput> inputs;
    inputs.reserve(100);
    for (int i = 0; i < 100; ++i) {
        stonesngems::GameParameters params = stonesngems::kDefaultGameParams;
        params["game_board_str"] = stonesngems::GameParameter(board_str[i % board_str.size()]);
        params["gravity"] = stonesngems::GameParameter(false);
        stonesngems::RNDGameState state(params);

        // Alternating threads get alternating models
        if (i % 2 == 0) {
            inputs.push_back({i, state, evaluator_A.get() });
        } else {
            inputs.push_back({i, state, evaluator_B.get() });
        }
    }

    std::vector<bool> results = pool.run(search, inputs);
}