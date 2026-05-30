#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace paged_kv {

struct KVCacheConfig {
    int num_layers = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int block_size = 16;
    int max_blocks = 0;
    int dtype_bytes = 2;

    size_t bytes_per_block() const {
        return static_cast<size_t>(2) * num_layers * block_size * num_kv_heads *
               head_dim * dtype_bytes;
    }

    size_t total_bytes() const {
        return static_cast<size_t>(max_blocks) * bytes_per_block();
    }
};

struct SequenceState {
    int length = 0;
    std::vector<int> block_table;
};

class KVCacheManager {
public:
    explicit KVCacheManager(KVCacheConfig config) : config_(config) {
        validate_config();
        free_blocks_.reserve(config_.max_blocks);
        for (int block = config_.max_blocks - 1; block >= 0; --block) {
            free_blocks_.push_back(block);
        }
    }

    void create_sequence(const std::string& request_id) {
        if (sequences_.count(request_id) != 0) {
            throw std::runtime_error("sequence already exists");
        }
        sequences_[request_id] = SequenceState{};
    }

    void reserve_tokens(const std::string& request_id, int target_length) {
        if (target_length < 0) {
            throw std::invalid_argument("target_length must be non-negative");
        }
        SequenceState& sequence = sequence_state(request_id);
        const int required_blocks =
            (target_length + config_.block_size - 1) / config_.block_size;
        while (static_cast<int>(sequence.block_table.size()) < required_blocks) {
            sequence.block_table.push_back(allocate_block());
        }
        if (target_length > sequence.length) {
            sequence.length = target_length;
        }
    }

    std::vector<int> free_sequence(const std::string& request_id) {
        SequenceState sequence = sequence_state(request_id);
        sequences_.erase(request_id);
        for (int block : sequence.block_table) {
            free_blocks_.push_back(block);
        }
        return sequence.block_table;
    }

    const std::vector<int>& block_table(const std::string& request_id) const {
        return sequence_state(request_id).block_table;
    }

    int length(const std::string& request_id) const {
        return sequence_state(request_id).length;
    }

    int available_blocks() const {
        return static_cast<int>(free_blocks_.size());
    }

    const KVCacheConfig& config() const {
        return config_;
    }

private:
    int allocate_block() {
        if (free_blocks_.empty()) {
            throw std::runtime_error("paged KV cache OOM: no free blocks");
        }
        const int block = free_blocks_.back();
        free_blocks_.pop_back();
        return block;
    }

    SequenceState& sequence_state(const std::string& request_id) {
        auto it = sequences_.find(request_id);
        if (it == sequences_.end()) {
            throw std::runtime_error("unknown sequence");
        }
        return it->second;
    }

    const SequenceState& sequence_state(const std::string& request_id) const {
        auto it = sequences_.find(request_id);
        if (it == sequences_.end()) {
            throw std::runtime_error("unknown sequence");
        }
        return it->second;
    }

    void validate_config() const {
        if (config_.num_layers <= 0 || config_.num_kv_heads <= 0 ||
            config_.head_dim <= 0 || config_.block_size <= 0 ||
            config_.max_blocks <= 0 || config_.dtype_bytes <= 0) {
            throw std::invalid_argument("invalid KV cache config");
        }
    }

    KVCacheConfig config_;
    std::vector<int> free_blocks_;
    std::unordered_map<std::string, SequenceState> sequences_;
};

}  // namespace paged_kv
