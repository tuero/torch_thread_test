#include "search.h"

#include <cmath>
#include <memory>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

const int ALLOCATE_INCREMENT = 2000;
const int BUDGET_NODES = 2000;

// Holds grounded state memory, nodes point to a state in this set (reduce duplicate memory)
struct GroundedStateHash {
    std::size_t operator()(const RNDGameState *state) const {
        return state->get_hash();
    }
};

struct GroundedStateCompareEqual {
    bool operator()(const RNDGameState *left, const RNDGameState *right) const {
        return *(left) == *(right);
    }
};

// Node used in search
struct Node {
    Node(Node *parent, const RNDGameState *state, double p, double g, double cost, int action)
        : parent(parent), state(state), p(p), g(g), levin_cost(cost), action(action) {}

    void set_values(Node *parent_, const RNDGameState *state_, double p_, double g_, int action_) {
        parent = parent_;
        state = state_;
        p = p_;
        g = g_;
        action = action_;
    }

    Node *parent;
    const RNDGameState *state;
    double p;
    double g;
    double levin_cost = 0;
    int action = -1;
    double h = 0;
    std::vector<double> action_log_policy;
};

// Compare function for nodes (f-cost, then g-cost on tiebreaks)
struct NodeCompareOrdered {
    bool operator()(const Node *left, const Node *right) const {
        return left->levin_cost > right->levin_cost || (left->levin_cost == right->levin_cost && left->g > right->g);
    }
};

// Node hash (state hash)
struct NodeHash {
    std::size_t operator()(const Node *node) const {
        return node->state->get_hash();
    }
};

// Node equality compare (underlying state) for closed
struct NodeCompareEqual {
    bool operator()(const Node *left, const Node *right) const {
        return *(left->state) == *(right->state);
    }
};

// Take log of policy and apply noise
std::vector<double> log_policy_noise(const std::vector<double> &policy, double epsilon = 0) {
    std::vector<double> log_policy;
    double noise = 1.0 / policy.size();
    for (const auto p : policy) {
        log_policy.push_back(std::log(((1.0 - epsilon) * p) + (epsilon * noise) + 1e-8));
    }
    return log_policy;
}

// Holds block allocation of states
struct StateContainer {
    using GroundedStateSet = std::unordered_set<const RNDGameState *, GroundedStateHash, GroundedStateCompareEqual>;
    StateContainer() = delete;
    StateContainer(const RNDGameState &state)
        : init_state(state), states(1, std::vector<RNDGameState>(ALLOCATE_INCREMENT, state)) {}

    void add_state(const RNDGameState &state) {
        assert(idx < (int)states.back().size() - 1);
        if (has_state(state)) {
            return;
        }

        states.back()[++idx] = state;
        state_set.insert(&states.back()[idx]);

        // Init next block of memory
        if (idx >= ALLOCATE_INCREMENT - 1) {
            states.emplace_back(std::vector<RNDGameState>(ALLOCATE_INCREMENT, init_state));
            idx = -1;
        }
    }

    bool has_state(const RNDGameState &state) {
        return state_set.find(&state) != state_set.end();
    }

    const RNDGameState *get_state(const RNDGameState &state) {
        auto itr = state_set.find(&state);
        return (itr == state_set.end()) ? nullptr : *itr;
    }

    RNDGameState init_state;
    std::vector<std::vector<RNDGameState>> states;
    GroundedStateSet state_set;
    int idx = -1;
};

struct NodeBuffer {
    Node *get_node() {
        if (idx + 1 >= (int)nodes.size()) {
            nodes.reserve(nodes.size() + ALLOCATE_INCREMENT);
            for (int i = 0; i < ALLOCATE_INCREMENT; ++i) {
                nodes.emplace_back(std::make_unique<Node>(nullptr, nullptr, 0, 0, 0, -1));
            }
        }
        return nodes[++idx].get();
    }
    std::vector<std::unique_ptr<Node>> nodes;
    int idx = -1;
};

// Cost function for PHS*, generalized LevinTS (if predicted_h = 0)
double phs_cost(const Node *node, double predicted_h) {
    predicted_h = (predicted_h < 0) ? 0 : predicted_h;
    return std::log(predicted_h + node->g + 1e-8) - (node->p * (1.0 + (predicted_h / node->g)));
}

bool search(const SearchInput &input) {
    using NodePointer = Node *;
    int expanded = 0;

    ModelEvaluator *model_eval = input.model_evaluator;
    std::vector<Observation> inference_inputs{input.state.get_observation()};
    InferenceOutput pred = model_eval->Inference(inference_inputs)[0];

    // Pre-allocate memory
    StateContainer state_buffer(input.state);
    NodeBuffer node_buffer;

    RNDGameState root_state = input.state;
    NodePointer root_node = node_buffer.get_node();
    root_node->set_values(nullptr, &root_state, 0, 0, -1);
    root_node->action_log_policy = log_policy_noise(pred.policy);

    state_buffer.add_state(root_state);

    std::priority_queue<NodePointer, std::vector<NodePointer>, NodeCompareOrdered> open;
    std::unordered_set<NodePointer, NodeHash, NodeCompareEqual> closed;
    open.push(root_node);

    std::vector<NodePointer> children_to_predict;
    std::vector<Observation> child_inference_inputs;
    while (!open.empty()) {
        NodePointer node = open.top();
        open.pop();
        closed.insert(node);
        ++expanded;

        // Solution found
        if (node->state->is_solution()) {
            return true;
        }

        // Timeout
        if (expanded >= BUDGET_NODES) {
            break;
        }

        const std::vector<int> actions = node->state->legal_actions();

        // Consider all children
        assert(actions.size() == node->action_log_policy.size());
        for (int i = 0; i < (int)actions.size(); ++i) {    // Buffer empty, reallocate
            RNDGameState child_state = *node->state;
            child_state.apply_action(actions[i]);

            // If terminal i.e. condition not met, then don't add for inference
            if (child_state.is_terminal()) {
                continue;
            }

            state_buffer.add_state(child_state);
            NodePointer child_node = node_buffer.get_node();
            child_node->set_values(node, state_buffer.get_state(child_state), node->p + node->action_log_policy[i],
                                   node->g + 1, actions[i]);

            // We will batch predict
            children_to_predict.push_back(child_node);
            child_inference_inputs.push_back(child_node->state->get_observation());
        }

        // Enough children saved to batch inference
        if ((int)children_to_predict.size() >= 32 || open.empty()) {
            std::vector<InferenceOutput> predictions = model_eval->Inference(child_inference_inputs);
            for (int i = 0; i < (int)predictions.size(); ++i) {
                NodePointer child_node = children_to_predict[i];
                if (closed.find(child_node) == closed.end()) {
                    const InferenceOutput &pred = predictions[i];
                    child_node->action_log_policy = log_policy_noise(pred.policy);
                    child_node->levin_cost = phs_cost(child_node, pred.heuristic);
                    child_node->h = pred.heuristic;
                    open.push(child_node);
                }
            }
            children_to_predict.clear();
            child_inference_inputs.clear();
        }
    }

    return false;
}
