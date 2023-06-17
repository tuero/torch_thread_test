#ifndef STONESNGEMS_BASE_H
#define STONESNGEMS_BASE_H

#include <array>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "definitions.h"
#include "util.h"

namespace stonesngems {

// Game parameter can be boolean, integral or floating point
using GameParameter = std::variant<bool, int, float, std::string>;
using GameParameters = std::unordered_map<std::string, GameParameter>;

// Default game parameters
static const GameParameters kDefaultGameParams{
    {"obs_show_ids", GameParameter(false)},       // Flag to show object ids in observation instead of binary channels
    {"magic_wall_steps", GameParameter(140)},     // Number of steps before magic wall expire
    {"blob_chance", GameParameter(20)},           // Chance to spawn another blob (out of 256)
    {"blob_max_percentage",
     GameParameter(static_cast<float>(0.16))},    // Max number of blobs before they collapse (percentage of map size)
    {"rng_seed", GameParameter(0)},               // Seed for anything that uses the rng
    {"game_board_str", GameParameter(std::string("2|2|-1|0|0|1|1|8"))},    // Game board string
    {"gravity", GameParameter(true)},                                      // Game board string
    {"blob_swap", GameParameter(-1)},                                      // Blob swap hidden element
};

// Shared global state information relevant to all states for the given game
struct SharedStateInfo {
    SharedStateInfo(const GameParameters &params)
        : params(params),
          obs_show_ids(std::get<bool>(params.at("obs_show_ids"))),
          magic_wall_steps(std::get<int>(params.at("magic_wall_steps"))),
          blob_chance(std::get<int>(params.at("blob_chance"))),
          blob_max_size(0),
          blob_max_percentage(std::get<float>(params.at("blob_max_percentage"))),
          rng_seed(std::get<int>(params.at("rng_seed"))),
          game_board_str(std::get<std::string>(params.at("game_board_str"))),
          gravity(std::get<bool>(params.at("gravity"))) {}
    GameParameters params;                      // Copy of game parameters for state resetting
    bool obs_show_ids;                          // Flag to show object IDs (currently not used)
    uint16_t magic_wall_steps;                  // Number of steps the magic wall stays active for
    uint8_t blob_chance;                        // Chance (out of 256) for blob to spawn
    uint16_t blob_max_size;                     // Max blob size in terms of grid spaces
    float blob_max_percentage;                  // Max blob size as percentage of map size
    int rng_seed;                               // Seed
    std::string game_board_str;                 // String representation of the starting state
    bool gravity;                               // Flag if gravity is on, affects stones/gems
    std::unordered_map<int, uint64_t> zrbht;    // Zobrist hashing table
    std::vector<bool> in_bounds_board;          // Fast check for single-step in bounds
    std::vector<int> board_to_inbounds;         // Indexing conversion for in bounds checking
};

// Information specific for the current game state
struct LocalState {
    LocalState()
        : magic_wall_steps(0),
          blob_size(0),
          blob_swap(-1),
          gems_collected(0),
          current_reward(0),
          reward_signal(0),
          magic_active(false),
          blob_enclosed(true),
          steps_remaining(-1),
          random_state(1),
          id_state(0) {}

    bool operator==(const LocalState &other) const {
        return magic_wall_steps == other.magic_wall_steps && blob_size == other.blob_size &&
               gems_collected == other.gems_collected && magic_active == other.magic_active &&
               blob_enclosed == other.blob_enclosed;
    }
    using id_type = uint16_t;

    uint16_t magic_wall_steps;                        // Number of steps remaining for the magic wall
    uint16_t blob_size;                               // Current size of the blob
    int8_t blob_swap;                                 // Swap element when the blob vanishes
    uint8_t gems_collected;                           // Number of gems collected
    uint8_t current_reward;                           // Reward for the current game state
    uint64_t reward_signal;                           // Signal for external information about events
    bool magic_active;                                // Flag if magic wall is currently active
    bool blob_enclosed;                               // Flag if blob is enclosed
    int steps_remaining;                              // Number of steps remaining (if timeout set)
    uint64_t random_state;                            // State of Xorshift rng
    uint16_t id_state;                                // Current ID state
    std::unordered_map<int, id_type> index_id_map;    // Index to ID mapping
    std::unordered_map<id_type, int> id_index_map;    // ID to index mapping
};

// Game state
class RNDGameState {
public:
    RNDGameState(const GameParameters &params = kDefaultGameParams);

    bool operator==(const RNDGameState &other) const;
    bool operator!=(const RNDGameState &other) const;

    /**
     * Reset the environment to the state as given by the GameParameters
     */
    void reset();

    /**
     * Apply the action to the current state, and set the reward and signals.
     * @param action The action to apply, should be one of the legal actions
     */
    void apply_action(int action);

    /**
     * Check if the state is terminal, meaning either solution, timeout, or agent dies.
     * @return True if terminal, false otherwise
     */
    bool is_terminal() const;

    /**
     * Check if the state is in the solution state (agent inside exit).
     * @return True if terminal, false otherwise
     */
    bool is_solution() const;

    /**
     * Get the legal actions which can be applied in the state.
     * @return vector containing each actions available
     */
    std::vector<int> legal_actions() const;

    /**
     * Get the shape the observations should be viewed as.
     * @return vector indicating observation CHW
     */
    std::array<int, 3> observation_shape() const;

    /**
     * Get a flat representation of the current state observation.
     * The observation should be viewed as the shape given by observation_shape().
     * @return vector where 1 represents object at position
     */
    std::vector<float> get_observation() const;

    /**
     * Get the index corresponding to the given position
     * @return the flat index
     */
    int position_to_index(const std::pair<int, int> &position) const;

    /**
     * Get the position corresponding to the given index
     * @return The position
     */
    std::pair<int, int> index_to_position(int index) const;

    /**
     * Get the flat (HWC) image representation of the current state
     * @return flattened byte vector represending RGB values (HWC)
     */
    std::vector<uint8_t> to_image() const;

    static std::vector<uint8_t> board_to_image(const std::vector<int8_t> &board, int rows, int cols);

    /**
     * Get the current reward signal as a result of the previous action taken.
     * @return bit field representing events that occured
     */
    uint64_t get_reward_signal() const;

    /**
     * Get the hash representation for the current state.
     * @return hash value
     */
    uint64_t get_hash() const;

    /**
     * Get all positions for a given element type
     * @param element The hidden cell type of the element to search for
     * @return pair of positions for each instance of element
     */
    std::vector<std::pair<int, int>> get_positions(HiddenCellType element) const;

    /**
     * Get all indices for a given element type
     * @param element The hidden cell type of the element to search for
     * @return flat indices for each instance of element
     */
    std::vector<int> get_indices(HiddenCellType element) const;

    /**
     * Check if a given position is in bounds
     * @param position The position to check
     * @return True if the position is in bounds, false otherwise
     */
    bool is_pos_in_bounds(const std::pair<int, int> &position) const;

    /**
     * Get the ID of the element at the given flat index
     * @param index The index to query
     * @return the ID of the element if trackable, else -1
     */
    int get_index_id(int index) const;

    /**
     * Get the index of the element for the given ID
     * @param id The id to query
     * @return the flat index of the element if trackable, else -1
     */
    int get_id_index(int id) const;

    /**
     * Get all possible reward codes from the current state
     * @return set of possible reward codes from the state
     */
    std::unordered_set<RewardCodes> get_valid_rewards() const;

    /**
     * Get the agent index position, or code if in exit (solution) or dead (failure)
     * @return Agent index, -1 if in exit (solution), or -2 if dead
     */
    int get_agent_pos() const;

    /**
     * Get the agent index position, even if in exit
     * @return Agent index
     */
    int get_agent_index() const;

    /**
     * Get the hidden cell type ID at the given index
     * @param index The index to query
     * @return The cell type ID
     */
    int8_t get_index_item(int index) const;

    /**
     * Get the hidden cell item at the given index
     */
    HiddenCellType get_hidden_item(int index) const;

    friend std::ostream &operator<<(std::ostream &os, const RNDGameState &state);

private:
    int IndexFromAction(int index, int action) const;
    int BoundsIndexFromAction(int index, int action) const;
    bool InBounds(int index, int action = Directions::kNoop) const;
    bool IsType(int index, const Element &element, int action = Directions::kNoop) const;
    bool HasProperty(int index, int property, int action = Directions::kNoop) const;
    void UpdateIDIndex(int index_old, int index_new);
    void UpdateIndexID(int index);
    void AddIndexID(int index);
    void RemoveIndexID(int index);
    void MoveItem(int index, int action);
    void SetItem(int index, const Element &element, int id, int action = Directions::kNoop);
    const Element &GetItem(int index, int action = Directions::kNoop) const;
    bool IsTypeAdjacent(int index, const Element &element) const;

    bool CanRollLeft(int index) const;
    bool CanRollRight(int index) const;
    void RollLeft(int index, const Element &element);
    void RollRight(int index, const Element &element);
    void Push(int index, const Element &stationary, const Element &falling, int action);
    void MoveThroughMagic(int index, const Element &element);
    void Explode(int index, const Element &element, int action = Directions::kNoop);

    void UpdateStone(int index);
    void UpdateStoneFalling(int index);
    void UpdateDiamond(int index);
    void UpdateDiamondFalling(int index);
    void UpdateNut(int index);
    void UpdateNutFalling(int index);
    void UpdateBomb(int index);
    void UpdateBombFalling(int index);
    void UpdateExit(int index);
    void UpdateAgent(int index, int action);
    void UpdateFirefly(int index, int action);
    void UpdateButterfly(int index, int action);
    void UpdateOrange(int index, int action);
    void UpdateMagicWall(int index);
    void UpdateBlob(int index);
    void UpdateExplosions(int index);
    void OpenGate(const Element &element);

    void StartScan();
    void EndScan();

    std::shared_ptr<SharedStateInfo> shared_state_ptr;
    Board board;
    LocalState local_state;
};

}    // namespace stonesngems

#endif    // STONESNGEMS_BASE_H
