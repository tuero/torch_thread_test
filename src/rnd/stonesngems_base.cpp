#include "stonesngems_base.h"

#include <cstdint>

#include "definitions.h"

namespace stonesngems {

RNDGameState::RNDGameState(const GameParameters &params)
    : shared_state_ptr(std::make_shared<SharedStateInfo>(params)),
      board(util::parse_board_str(std::get<std::string>(params.at("game_board_str")))) {
    reset();
}

bool RNDGameState::operator==(const RNDGameState &other) const {
    return local_state == other.local_state && board == other.board;
}

bool RNDGameState::operator!=(const RNDGameState &other) const {
    return !(*this == other);
}

// ---------------------------------------------------------------------------

// https://en.wikipedia.org/wiki/Xorshift
// Portable RNG Seed
uint64_t splitmix64(uint64_t seed) {
    uint64_t result = seed + 0x9E3779B97f4A7C15;
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return result ^ (result >> 31);
}

// Portable RNG
uint64_t xorshift64(uint64_t &s) {
    uint64_t x = s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    s = x;
    return x;
}

void RNDGameState::reset() {
    // Board, local, and shared state info
    board = util::parse_board_str(shared_state_ptr->game_board_str);
    local_state = LocalState();
    local_state.random_state = splitmix64(shared_state_ptr->rng_seed);
    local_state.steps_remaining = board.max_steps;
    shared_state_ptr->blob_chance = (board.cols * board.rows) * shared_state_ptr->blob_max_size;

    // Set the item IDs
    for (int i = 0; i < board.cols * board.rows; ++i) {
        AddIndexID(i);
    }

    // zorbist hashing
    std::mt19937 gen(shared_state_ptr->rng_seed);
    std::uniform_int_distribution<uint64_t> dist(0);
    for (int channel = 0; channel < kNumHiddenCellType; ++channel) {
        for (int i = 0; i < board.cols * board.rows; ++i) {
            shared_state_ptr->zrbht[(channel * board.cols * board.rows) + i] = dist(gen);
        }
    }

    // Set initial hash
    for (int i = 0; i < board.cols * board.rows; ++i) {
        board.zorb_hash ^= shared_state_ptr->zrbht.at((board.item(i) * board.cols * board.rows) + i);
    }

    // In bounds fast access
    shared_state_ptr->in_bounds_board.clear();
    shared_state_ptr->in_bounds_board.insert(shared_state_ptr->in_bounds_board.end(),
                                             (board.cols + 2) * (board.rows + 2), true);
    // Pad the outer boarder
    for (int i = 0; i < board.cols + 2; ++i) {
        shared_state_ptr->in_bounds_board[i] = false;
        shared_state_ptr->in_bounds_board[(board.rows + 1) * (board.cols + 2) + i] = false;
    }
    for (int i = 0; i < board.rows + 2; ++i) {
        shared_state_ptr->in_bounds_board[i * (board.cols + 2)] = false;
        shared_state_ptr->in_bounds_board[i * (board.cols + 2) + board.cols + 1] = false;
    }
    // In bounds idx conversion table
    shared_state_ptr->board_to_inbounds.clear();
    for (int r = 0; r < board.rows; ++r) {
        for (int c = 0; c < board.cols; ++c) {
            shared_state_ptr->board_to_inbounds.push_back((board.cols + 2) * (r + 1) + c + 1);
        }
    }
}

void RNDGameState::apply_action(int action) {
    assert(action >= 0 && action < kNumActions);
    StartScan();

    // Handle agent first
    UpdateAgent(board.agent_idx, static_cast<Directions>(action));

    // Handle all other items
    for (int i = 0; i < board.rows * board.cols; ++i) {
        if (board.has_updated[i]) {    // Item already updated
            continue;
        }
        switch (board.item(i)) {
            // Handle non-compound types
            case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kStone):
                UpdateStone(i);
                break;
            case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kStoneFalling):
                UpdateStoneFalling(i);
                break;
            case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kDiamond):
                UpdateDiamond(i);
                break;
            case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kDiamondFalling):
                UpdateDiamondFalling(i);
                break;
            case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kNut):
                UpdateNut(i);
                break;
            case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kNutFalling):
                UpdateNutFalling(i);
                break;
            case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kBomb):
                UpdateBomb(i);
                break;
            case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kBombFalling):
                UpdateBombFalling(i);
                break;
            case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kExitClosed):
                UpdateExit(i);
                break;
            case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kBlob):
                UpdateBlob(i);
                break;
            default:
                // Handle compound types
                const Element &element = kCellTypeToElement[board.item(i) + 1];
                if (IsButterfly(element)) {
                    UpdateButterfly(i, kButterflyToDirection.at(element));
                } else if (IsFirefly(element)) {
                    UpdateFirefly(i, kFireflyToDirection.at(element));
                } else if (IsOrange(element)) {
                    UpdateOrange(i, kOrangeToDirection.at(element));
                } else if (IsMagicWall(element)) {
                    UpdateMagicWall(i);
                } else if (IsExplosion(element)) {
                    UpdateExplosions(i);
                }
                break;
        }
    }

    EndScan();
}

bool RNDGameState::is_terminal() const {
    // timeout or agent is either dead/in exit
    bool out_of_time = (board.max_steps > 0 && local_state.steps_remaining <= 0);
    return out_of_time || board.agent_pos < 0;
}

bool RNDGameState::is_solution() const {
    // not timeout and agent is in exit
    bool out_of_time = (board.max_steps > 0 && local_state.steps_remaining <= 0);
    return !out_of_time && board.agent_pos == kAgentPosExit;
}

std::vector<int> RNDGameState::legal_actions() const {
    return {Directions::kNoop, Directions::kUp, Directions::kRight, Directions::kDown, Directions::kLeft};
}

std::array<int, 3> RNDGameState::observation_shape() const {
    return {kNumVisibleCellType, board.cols, board.rows};
}

std::vector<float> RNDGameState::get_observation() const {
    int channel_length = board.cols * board.rows;
    std::vector<float> obs(kNumVisibleCellType * channel_length, 0);
    for (int i = 0; i < channel_length; ++i) {
        obs[static_cast<std::underlying_type_t<VisibleCellType>>(GetItem(i).visible_type) * channel_length + i] = 1;
    }
    return obs;
}

std::vector<uint8_t> RNDGameState::board_to_image(const std::vector<int8_t> &board, int rows, int cols) {
    int flat_size = cols * rows;
    std::vector<uint8_t> img(flat_size * 32 * 32 * 3, 0);
    for (int h = 0; h < rows; ++h) {
        for (int w = 0; w < cols; ++w) {
            int img_idx_top_left = h * (32 * 32 * 3 * cols) + (w * 32 * 3);
            VisibleCellType el = static_cast<VisibleCellType>(board[h * cols + w]);
            const std::vector<uint8_t> &data = img_asset_map.at(el);
            for (int r = 0; r < 32; ++r) {
                for (int c = 0; c < 32; ++c) {
                    int data_idx = (r * 3 * 32) + (3 * c);
                    int img_idx = (r * 32 * 3 * cols) + (3 * c) + img_idx_top_left;
                    img[img_idx + 0] = data[data_idx + 0];
                    img[img_idx + 1] = data[data_idx + 1];
                    img[img_idx + 2] = data[data_idx + 2];
                }
            }
        }
    }
    return img;
}

std::vector<uint8_t> RNDGameState::to_image() const {
    int flat_size = board.cols * board.rows;
    std::vector<uint8_t> img(flat_size * 32 * 32 * 3, 0);
    for (int h = 0; h < board.rows; ++h) {
        for (int w = 0; w < board.cols; ++w) {
            int img_idx_top_left = h * (32 * 32 * 3 * board.cols) + (w * 32 * 3);
            const std::vector<uint8_t> &data = img_asset_map.at(GetItem(h * board.cols + w).visible_type);
            for (int r = 0; r < 32; ++r) {
                for (int c = 0; c < 32; ++c) {
                    int data_idx = (r * 3 * 32) + (3 * c);
                    int img_idx = (r * 32 * 3 * board.cols) + (3 * c) + img_idx_top_left;
                    img[img_idx + 0] = data[data_idx + 0];
                    img[img_idx + 1] = data[data_idx + 1];
                    img[img_idx + 2] = data[data_idx + 2];
                }
            }
        }
    }
    return img;
}

uint64_t RNDGameState::get_reward_signal() const {
    return local_state.reward_signal;
}

uint64_t RNDGameState::get_hash() const {
    return board.zorb_hash;
}

std::vector<std::pair<int, int>> RNDGameState::get_positions(HiddenCellType element) const {
    std::vector<std::pair<int, int>> indices;
    for (const auto &idx : board.find_all(static_cast<std::underlying_type_t<HiddenCellType>>(element))) {
        indices.push_back({idx / board.cols, idx % board.cols});
    }
    return indices;
}

int RNDGameState::position_to_index(const std::pair<int, int> &position) const {
    return position.first * board.cols + position.second;
}

std::pair<int, int> RNDGameState::index_to_position(int index) const {
    return {index / board.cols, index % board.cols};
}

std::vector<int> RNDGameState::get_indices(HiddenCellType element) const {
    std::vector<int> indices;
    for (const auto &idx : board.find_all(static_cast<std::underlying_type_t<HiddenCellType>>(element))) {
        indices.push_back(idx);
    }
    return indices;
}

bool RNDGameState::is_pos_in_bounds(const std::pair<int, int> &position) const {
    return position.first >= 0 && position.second >= 0 && position.first < board.rows && position.second < board.cols;
}

int RNDGameState::get_index_id(int index) const {
    auto iter = local_state.index_id_map.find(index);
    return iter == local_state.index_id_map.end() ? -1 : static_cast<int>(iter->second);
}

int RNDGameState::get_id_index(int id) const {
    auto iter = local_state.id_index_map.find(static_cast<LocalState::id_type>(id));
    return iter == local_state.id_index_map.end() ? -1 : iter->second;
}

std::unordered_set<RewardCodes> RNDGameState::get_valid_rewards() const {
    std::unordered_set<RewardCodes> reward_codes;
    for (int i = 0; i < (int)board.grid.size(); ++i) {
        HiddenCellType el = static_cast<HiddenCellType>(board.grid[i]);
        if (kElementToRewardMap.find(el) != kElementToRewardMap.end()) {
            reward_codes.insert(kElementToRewardMap.at(el));
        }
    }
    return reward_codes;
}

int RNDGameState::get_agent_pos() const {
    return board.agent_pos;
}

int RNDGameState::get_agent_index() const {
    return board.agent_idx;
}

int8_t RNDGameState::get_index_item(int index) const {
    return board.item(index);
}

HiddenCellType RNDGameState::get_hidden_item(int index) const {
    return static_cast<HiddenCellType>(board.item(index));
}

std::ostream &operator<<(std::ostream &os, const RNDGameState &state) {
    for (int h = 0; h < state.board.rows; ++h) {
        for (int w = 0; w < state.board.cols; ++w) {
            os << kCellTypeToElement[state.board.grid[h * state.board.cols + w] + 1].id;
        }
        os << std::endl;
    }
    return os;
}

// ---------------------------------------------------------------------------

// Not safe, assumes InBounds has been called (or used in conjunction)
int RNDGameState::IndexFromAction(int index, int action) const {
    switch (action) {
        case Directions::kNoop:
            return index;
        case Directions::kUp:
            return index - board.cols;
        case Directions::kRight:
            return index + 1;
        case Directions::kDown:
            return index + board.cols;
        case Directions::kLeft:
            return index - 1;
        case Directions::kUpRight:
            return index - board.cols + 1;
        case Directions::kDownRight:
            return index + board.cols + 1;
        case Directions::kUpLeft:
            return index - board.cols - 1;
        case Directions::kDownLeft:
            return index + board.cols - 1;
        default:
            __builtin_unreachable();
    }
}
int RNDGameState::BoundsIndexFromAction(int index, int action) const {
    switch (action) {
        case Directions::kNoop:
            return index;
        case Directions::kUp:
            return index - (board.cols + 2);
        case Directions::kRight:
            return index + 1;
        case Directions::kDown:
            return index + (board.cols + 2);
        case Directions::kLeft:
            return index - 1;
        case Directions::kUpRight:
            return index - (board.cols + 2) + 1;
        case Directions::kDownRight:
            return index + (board.cols + 2) + 1;
        case Directions::kUpLeft:
            return index - (board.cols + 2) - 1;
        case Directions::kDownLeft:
            return index + (board.cols + 2) - 1;
        default:
            __builtin_unreachable();
    }
}

bool RNDGameState::InBounds(int index, int action) const {
    return shared_state_ptr->in_bounds_board[BoundsIndexFromAction(shared_state_ptr->board_to_inbounds[index], action)];
}

bool RNDGameState::IsType(int index, const Element &element, int action) const {
    int new_index = IndexFromAction(index, action);
    return InBounds(index, action) && GetItem(new_index) == element;
}

bool RNDGameState::HasProperty(int index, int property, int action) const {
    int new_index = IndexFromAction(index, action);
    return InBounds(index, action) && ((GetItem(new_index).properties & property) > 0);
}

void RNDGameState::UpdateIDIndex(int index_old, int index_new) {
    if (local_state.index_id_map.find(index_old) != local_state.index_id_map.end()) {
        auto id = local_state.index_id_map.at(index_old);
        local_state.index_id_map.erase(index_old);
        local_state.index_id_map[index_new] = id;
        local_state.id_index_map[id] = index_new;
    }
}

void RNDGameState::UpdateIndexID(int index) {
    if (local_state.index_id_map.find(index) != local_state.index_id_map.end()) {
        auto id_old = local_state.index_id_map.at(index);
        auto id_new = ++local_state.id_state;
        local_state.id_index_map.erase(id_old);
        local_state.id_index_map[id_new] = index;
        local_state.index_id_map[index] = id_new;
    }
}

void RNDGameState::AddIndexID(int index) {
    switch (board.item(index)) {
        case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kStone):
        case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kStoneFalling):
        case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kDiamond):
        case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kDiamondFalling):
        case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kNut):
        case static_cast<std::underlying_type_t<HiddenCellType>>(HiddenCellType::kNutFalling): {
            auto id = ++local_state.id_state;
            local_state.id_index_map[id] = index;
            local_state.index_id_map[index] = id;
            break;
        }
        default:
            break;
    }
}

void RNDGameState::RemoveIndexID(int index) {
    if (local_state.index_id_map.find(index) != local_state.index_id_map.end()) {
        auto id = local_state.index_id_map.at(index);
        local_state.id_index_map.erase(id);
        local_state.index_id_map.erase(index);
    }
}

void RNDGameState::MoveItem(int index, int action) {
    int new_index = IndexFromAction(index, action);
    board.zorb_hash ^= shared_state_ptr->zrbht.at((board.item(new_index) * board.cols * board.rows) + new_index);
    board.item(new_index) = board.item(index);
    board.zorb_hash ^= shared_state_ptr->zrbht.at((board.item(new_index) * board.cols * board.rows) + new_index);
    // grid_.ids[new_index] = grid_.ids[index];

    board.zorb_hash ^= shared_state_ptr->zrbht.at((board.item(index) * board.cols * board.rows) + index);
    board.item(index) = ElementToItem(kElEmpty);
    board.zorb_hash ^= shared_state_ptr->zrbht.at((ElementToItem(kElEmpty) * board.cols * board.rows) + index);
    board.has_updated[new_index] = true;
    // grid_.ids[index] = ++id_counter_;

    // Update ID
    UpdateIDIndex(index, new_index);
}

void RNDGameState::SetItem(int index, const Element &element, int id, int action) {
    (void)id;
    int new_index = IndexFromAction(index, action);
    board.zorb_hash ^= shared_state_ptr->zrbht.at((board.item(new_index) * board.cols * board.rows) + new_index);
    board.item(new_index) = ElementToItem(element);
    board.zorb_hash ^= shared_state_ptr->zrbht.at((ElementToItem(element) * board.cols * board.rows) + new_index);
    // grid_.ids[new_index] = id;
    board.has_updated[new_index] = true;
}

const Element &RNDGameState::GetItem(int index, int action) const {
    int new_index = IndexFromAction(index, action);
    return kCellTypeToElement[board.item(new_index) + 1];
}

bool RNDGameState::IsTypeAdjacent(int index, const Element &element) const {
    return IsType(index, element, Directions::kUp) || IsType(index, element, Directions::kLeft) ||
           IsType(index, element, Directions::kDown) || IsType(index, element, Directions::kRight);
}

// ---------------------------------------------------------------------------

bool RNDGameState::CanRollLeft(int index) const {
    return HasProperty(index, ElementProperties::kRounded, Directions::kDown) &&
           IsType(index, kElEmpty, Directions::kLeft) && IsType(index, kElEmpty, Directions::kDownLeft);
}

bool RNDGameState::CanRollRight(int index) const {
    return HasProperty(index, ElementProperties::kRounded, Directions::kDown) &&
           IsType(index, kElEmpty, Directions::kRight) && IsType(index, kElEmpty, Directions::kDownRight);
}

void RNDGameState::RollLeft(int index, const Element &element) {
    SetItem(index, element, -1);
    MoveItem(index, Directions::kLeft);
}

void RNDGameState::RollRight(int index, const Element &element) {
    SetItem(index, element, -1);
    MoveItem(index, Directions::kRight);
}

void RNDGameState::Push(int index, const Element &stationary, const Element &falling, int action) {
    int new_index = IndexFromAction(index, action);
    // Check if same direction past element is empty so that theres room to push
    if (IsType(new_index, kElEmpty, action)) {
        // Check if the element will become stationary or falling
        int next_index = IndexFromAction(new_index, action);
        bool is_empty = IsType(next_index, kElEmpty, Directions::kDown);
        // Move item and set as falling or stationary
        MoveItem(new_index, action);
        SetItem(next_index, is_empty ? falling : stationary, -1);
        // Move the agent
        MoveItem(index, action);
        board.agent_pos = IndexFromAction(index, action);    // Assume only agent is pushing?
        board.agent_idx = IndexFromAction(index, action);    // Assume only agent is pushing?
    }
}

void RNDGameState::MoveThroughMagic(int index, const Element &element) {
    // Check if magic wall is still active
    if (local_state.magic_wall_steps <= 0) {
        return;
    }
    local_state.magic_active = true;
    int index_wall = IndexFromAction(index, Directions::kDown);
    int index_under_wall = IndexFromAction(index_wall, Directions::kDown);
    // Need to ensure cell below magic wall is empty (so item can pass through)
    if (IsType(index_under_wall, kElEmpty)) {
        SetItem(index, kElEmpty, -1);
        SetItem(index_under_wall, element, -1);
        UpdateIDIndex(index, index_under_wall);
    }
}

void RNDGameState::Explode(int index, const Element &element, int action) {
    int new_index = IndexFromAction(index, action);
    auto it = kElementToExplosion.find(GetItem(new_index));
    const Element &ex = (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second;
    if (GetItem(new_index) == kElAgent) {
        board.agent_pos = kAgentPosDie;
    }
    SetItem(new_index, element, -1);
    RemoveIndexID(new_index);
    // Recursively check all directions for chain explosions
    for (int dir = 0; dir < kNumDirections; ++dir) {
        if (dir == Directions::kNoop || !InBounds(new_index, dir)) {
            continue;
        }
        if (HasProperty(new_index, ElementProperties::kCanExplode, dir)) {
            Explode(new_index, ex, dir);
        } else if (HasProperty(new_index, ElementProperties::kConsumable, dir)) {
            SetItem(new_index, ex, -1, dir);
            if (GetItem(new_index, dir) == kElAgent) {
                board.agent_pos = kAgentPosDie;
            }
        }
    }
}

// ---------------------------------------------------------------------------

void RNDGameState::UpdateStone(int index) {
    // If no gravity, do nothing
    if (!shared_state_ptr->gravity) {
        return;
    }

    // Boulder falls if empty below
    if (IsType(index, kElEmpty, Directions::kDown)) {
        SetItem(index, kElStoneFalling, -1);
        UpdateStoneFalling(index);
    } else if (CanRollLeft(index)) {    // Roll left/right if possible
        RollLeft(index, kElStoneFalling);
    } else if (CanRollRight(index)) {
        RollRight(index, kElStoneFalling);
    }
}

void RNDGameState::UpdateStoneFalling(int index) {
    // Continue to fall as normal
    if (IsType(index, kElEmpty, Directions::kDown)) {
        MoveItem(index, Directions::kDown);
    } else if (HasProperty(index, ElementProperties::kCanExplode, Directions::kDown)) {
        // Falling stones can cause elements to explode
        auto it = kElementToExplosion.find(GetItem(index, Directions::kDown));
        Explode(index, (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second, Directions::kDown);
    } else if (IsType(index, kElWallMagicOn, Directions::kDown) ||
               IsType(index, kElWallMagicDormant, Directions::kDown)) {
        MoveThroughMagic(index, kMagicWallConversion.at(GetItem(index)));
    } else if (IsType(index, kElNut, Directions::kDown)) {
        // Falling on a nut, crack it open to reveal a diamond!
        SetItem(index, kElDiamond, -1, Directions::kDown);
        UpdateIndexID(IndexFromAction(index, Directions::kDown));
    } else if (IsType(index, kElNut, Directions::kDown)) {
        // Falling on a bomb, explode!
        auto it = kElementToExplosion.find(GetItem(index));
        Explode(index, (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second);
    } else if (CanRollLeft(index)) {    // Roll left/right
        RollLeft(index, kElStoneFalling);
    } else if (CanRollRight(index)) {
        RollRight(index, kElStoneFalling);
    } else {
        // Default options is for falling stones to become stationary
        SetItem(index, kElStone, -1);
    }
}

void RNDGameState::UpdateDiamond(int index) {
    // If no gravity, do nothing
    if (!shared_state_ptr->gravity) {
        return;
    }

    // Diamond falls if empty below
    if (IsType(index, kElEmpty, Directions::kDown)) {
        SetItem(index, kElDiamondFalling, -1);
        UpdateDiamondFalling(index);
    } else if (CanRollLeft(index)) {    // Roll left/right if possible
        RollLeft(index, kElDiamondFalling);
    } else if (CanRollRight(index)) {
        RollRight(index, kElDiamondFalling);
    }
}

void RNDGameState::UpdateDiamondFalling(int index) {
    // Continue to fall as normal
    if (IsType(index, kElEmpty, Directions::kDown)) {
        MoveItem(index, Directions::kDown);
    } else if (HasProperty(index, ElementProperties::kCanExplode, Directions::kDown) &&
               !IsType(index, kElBomb, Directions::kDown) && !IsType(index, kElBombFalling, Directions::kDown)) {
        // Falling diamonds can cause elements to explode (but not bombs)
        auto it = kElementToExplosion.find(GetItem(index, Directions::kDown));
        Explode(index, (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second, Directions::kDown);
    } else if (IsType(index, kElWallMagicOn, Directions::kDown) ||
               IsType(index, kElWallMagicDormant, Directions::kDown)) {
        MoveThroughMagic(index, kMagicWallConversion.at(GetItem(index)));
    } else if (CanRollLeft(index)) {    // Roll left/right
        RollLeft(index, kElDiamondFalling);
    } else if (CanRollRight(index)) {
        RollRight(index, kElDiamondFalling);
    } else {
        // Default options is for falling diamond to become stationary
        SetItem(index, kElDiamond, -1);
    }
}

void RNDGameState::UpdateNut(int index) {
    // If no gravity, do nothing
    if (!shared_state_ptr->gravity) {
        return;
    }

    // Nut falls if empty below
    if (IsType(index, kElEmpty, Directions::kDown)) {
        SetItem(index, kElNutFalling, -1);
        UpdateNutFalling(index);
    } else if (CanRollLeft(index)) {    // Roll left/right
        RollLeft(index, kElNutFalling);
    } else if (CanRollRight(index)) {
        RollRight(index, kElNutFalling);
    }
}

void RNDGameState::UpdateNutFalling(int index) {
    // Continue to fall as normal
    if (IsType(index, kElEmpty, Directions::kDown)) {
        MoveItem(index, Directions::kDown);
    } else if (CanRollLeft(index)) {    // Roll left/right
        RollLeft(index, kElNutFalling);
    } else if (CanRollRight(index)) {
        RollRight(index, kElNutFalling);
    } else {
        // Default options is for falling nut to become stationary
        SetItem(index, kElNut, -1);
    }
}

void RNDGameState::UpdateBomb(int index) {
    // If no gravity, do nothing
    if (!shared_state_ptr->gravity) {
        return;
    }

    // Bomb falls if empty below
    if (IsType(index, kElEmpty, Directions::kDown)) {
        SetItem(index, kElBombFalling, -1);
        UpdateBombFalling(index);
    } else if (CanRollLeft(index)) {    // Roll left/right
        RollLeft(index, kElBomb);
    } else if (CanRollRight(index)) {
        RollRight(index, kElBomb);
    }
}

void RNDGameState::UpdateBombFalling(int index) {
    // Continue to fall as normal
    if (IsType(index, kElEmpty, Directions::kDown)) {
        MoveItem(index, Directions::kDown);
    } else if (CanRollLeft(index)) {    // Roll left/right
        RollLeft(index, kElBombFalling);
    } else if (CanRollRight(index)) {
        RollRight(index, kElBombFalling);
    } else {
        // Default options is for bomb to explode if stopped falling
        auto it = kElementToExplosion.find(GetItem(index));
        Explode(index, (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second);
    }
}

void RNDGameState::UpdateExit(int index) {
    // Open exit if enough gems collected
    if (local_state.gems_collected >= board.gems_required) {
        SetItem(index, kElExitOpen, -1);
    }
}

void RNDGameState::UpdateAgent(int index, int action) {
    // If action results not in bounds, don't do anything
    if (!InBounds(index, action)) {
        return;
    }

    if (IsType(index, kElEmpty, action) || IsType(index, kElDirt, action)) {    // Move if empty/dirt
        MoveItem(index, action);
        board.agent_pos = IndexFromAction(index, action);
        board.agent_idx = IndexFromAction(index, action);
    } else if (IsType(index, kElDiamond, action) || IsType(index, kElDiamondFalling, action)) {    // Collect gems
        ++local_state.gems_collected;
        local_state.current_reward += kPointMap.at(GetItem(index, action).cell_type);
        local_state.reward_signal |= RewardCodes::kRewardCollectDiamond;
        MoveItem(index, action);
        RemoveIndexID(IndexFromAction(index, action));
        board.agent_pos = IndexFromAction(index, action);
        board.agent_idx = IndexFromAction(index, action);
    } else if (IsActionHorz(action) && HasProperty(index, ElementProperties::kPushable, action)) {
        // Push stone, nut, or bomb if action is horizontal
        Push(index, GetItem(index, action), kElToFalling.at(GetItem(index, action)), action);
    } else if (IsKey(GetItem(index, action))) {
        // Collecting key, set gate open
        Element key_type = GetItem(index, action);
        OpenGate(kKeyToGate.at(key_type));
        MoveItem(index, action);
        board.agent_pos = IndexFromAction(index, action);
        board.agent_idx = IndexFromAction(index, action);
        local_state.reward_signal |= RewardCodes::kRewardCollectKey;
        local_state.reward_signal |= kKeyToSignal.at(key_type);
    } else if (IsOpenGate(GetItem(index, action))) {
        // Walking through an open gate, with traversable element on other side
        int index_gate = IndexFromAction(index, action);
        if (HasProperty(index_gate, ElementProperties::kTraversable, action)) {
            // Correct for landing on traversable elements
            if (IsType(index_gate, kElDiamond, action) || IsType(index_gate, kElDiamondFalling, action)) {
                ++local_state.gems_collected;
                local_state.current_reward += kPointMap.at(GetItem(index_gate, action).cell_type);
                local_state.reward_signal |= RewardCodes::kRewardCollectDiamond;
            } else if (IsKey(GetItem(index_gate, action))) {
                Element key_type = GetItem(index_gate, action);
                OpenGate(kKeyToGate.at(key_type));
                local_state.reward_signal |= RewardCodes::kRewardCollectKey;
                local_state.reward_signal |= kKeyToSignal.at(key_type);
            }
            // Move agent through gate
            SetItem(index_gate, kElAgent, -1, action);
            SetItem(index, kElEmpty, -1);
            board.agent_pos = IndexFromAction(index_gate, action);
            board.agent_idx = IndexFromAction(index_gate, action);
            local_state.reward_signal |= RewardCodes::kRewardWalkThroughGate;
            local_state.reward_signal |= kGateToSignal.at(GetItem(index_gate));
        }
    } else if (IsType(index, kElExitOpen, action)) {
        // Walking into exit after collecting enough gems
        MoveItem(index, action);
        SetItem(index, kElAgentInExit, -1, action);
        board.agent_pos = kAgentPosExit;
        board.agent_idx = IndexFromAction(index, action);
        local_state.reward_signal |= RewardCodes::kRewardWalkThroughExit;
        local_state.current_reward += local_state.steps_remaining * 100 / board.max_steps;
    }
}

void RNDGameState::UpdateFirefly(int index, int action) {
    int new_dir = kRotateLeft[action];
    if (IsTypeAdjacent(index, kElAgent) || IsTypeAdjacent(index, kElBlob)) {
        // Explode if touching the agent/blob
        auto it = kElementToExplosion.find(GetItem(index));
        Explode(index, (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second);
    } else if (IsType(index, kElEmpty, new_dir)) {
        // Fireflies always try to rotate left, otherwise continue forward
        SetItem(index, kDirectionToFirefly[new_dir], -1);
        MoveItem(index, new_dir);
    } else if (IsType(index, kElEmpty, action)) {
        SetItem(index, kDirectionToFirefly[action], -1);
        MoveItem(index, action);
    } else {
        // No other options, rotate right
        SetItem(index, kDirectionToFirefly[kRotateRight[action]], -1);
    }
}

void RNDGameState::UpdateButterfly(int index, int action) {
    int new_dir = kRotateRight[action];
    if (IsTypeAdjacent(index, kElAgent) || IsTypeAdjacent(index, kElBlob)) {
        // Explode if touching the agent/blob
        auto it = kElementToExplosion.find(GetItem(index));
        Explode(index, (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second);
    } else if (IsType(index, kElEmpty, new_dir)) {
        // Butterflies always try to rotate right, otherwise continue forward
        SetItem(index, kDirectionToButterfly[new_dir], -1);
        MoveItem(index, new_dir);
    } else if (IsType(index, kElEmpty, action)) {
        SetItem(index, kDirectionToButterfly[action], -1);
        MoveItem(index, action);
    } else {
        // No other options, rotate right
        SetItem(index, kDirectionToButterfly[kRotateLeft[action]], -1);
    }
}

void RNDGameState::UpdateOrange(int index, int action) {
    if (IsType(index, kElEmpty, action)) {
        // Continue moving in direction
        MoveItem(index, action);
    } else if (IsTypeAdjacent(index, kElAgent)) {
        // Run into the agent, explode!
        auto it = kElementToExplosion.find(GetItem(index));
        Explode(index, (it == kElementToExplosion.end()) ? kElExplosionEmpty : it->second);
    } else {
        // Blocked, roll for new direction
        std::vector<int> open_dirs;
        for (int dir = 0; dir < kNumActions; ++dir) {
            if (dir == Directions::kNoop || !InBounds(index, dir)) {
                continue;
            }
            if (IsType(index, kElEmpty, dir)) {
                open_dirs.push_back(dir);
            }
        }
        // Roll available directions
        if (!open_dirs.empty()) {
            int new_dir = open_dirs[xorshift64(local_state.random_state) % open_dirs.size()];
            SetItem(index, kDirectionToOrange[new_dir], -1);
        }
    }
}

void RNDGameState::UpdateMagicWall(int index) {
    // Dorminant, active, then expired once time runs out
    if (local_state.magic_active) {
        SetItem(index, kElWallMagicOn, -1);
    } else if (local_state.magic_wall_steps > 0) {
        SetItem(index, kElWallMagicDormant, -1);
    } else {
        SetItem(index, kElWallMagicExpired, -1);
    }
}

void RNDGameState::UpdateBlob(int index) {
    // Replace blobs if swap element set
    if (local_state.blob_swap != ElementToItem(kNullElement)) {
        SetItem(index, kCellTypeToElement[local_state.blob_swap + 1], -1);
        AddIndexID(index);
        return;
    }
    ++local_state.blob_size;
    // Check if at least one tile blob can grow to
    if (IsTypeAdjacent(index, kElEmpty) || IsTypeAdjacent(index, kElDirt)) {
        local_state.blob_enclosed = false;
    }
    // Roll if to grow and direction
    bool will_grow = (xorshift64(local_state.random_state) % 256) < shared_state_ptr->blob_chance;
    int grow_dir = xorshift64(local_state.random_state) % kNumActions;
    if (will_grow && (IsType(index, kElEmpty, grow_dir) || IsType(index, kElDirt, grow_dir))) {
        SetItem(index, kElBlob, -1, grow_dir);
        // TODO test this
        RemoveIndexID(IndexFromAction(index, grow_dir));
    }
}

void RNDGameState::UpdateExplosions(int index) {
    SetItem(index, kExplosionToElement.at(GetItem(index)), -1);
    AddIndexID(index);
}

void RNDGameState::OpenGate(const Element &element) {
    std::vector<int> closed_gate_indices = board.find_all(ElementToItem(element));
    for (const auto &index : closed_gate_indices) {
        SetItem(index, kGateOpenMap.at(GetItem(index)), -1);
    }
}

// ---------------------------------------------------------------------------

void RNDGameState::StartScan() {
    if (local_state.steps_remaining > 0) {
        local_state.steps_remaining += -1;
    }
    local_state.current_reward = 0;
    local_state.blob_size = 0;
    local_state.blob_enclosed = true;
    local_state.reward_signal = 0;
    board.reset_updated();
}

void RNDGameState::EndScan() {
    if (local_state.blob_swap == ElementToItem(kNullElement)) {
        if (local_state.blob_enclosed) {
            local_state.blob_swap = ElementToItem(kElDiamond);
        }
        if (local_state.blob_size > shared_state_ptr->blob_max_size) {
            local_state.blob_swap = ElementToItem(kElStone);
        }
    }
    if (local_state.magic_active) {
        local_state.magic_wall_steps = std::max(local_state.magic_wall_steps - 1, 0);
    }
    local_state.magic_active = local_state.magic_active && (local_state.magic_wall_steps > 0);
}

}    // namespace stonesngems
