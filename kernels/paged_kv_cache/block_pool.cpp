#include <stdexcept>
#include <unordered_set>
#include <vector>

class PagedKVBlockPool {
public:
    explicit PagedKVBlockPool(int total_blocks) {
        if (total_blocks <= 0) {
            throw std::invalid_argument("total_blocks must be positive");
        }
        total_blocks_ = total_blocks;
        free_blocks_.reserve(total_blocks);
        for (int block = total_blocks - 1; block >= 0; --block) {
            free_blocks_.push_back(block);
            free_set_.insert(block);
        }
    }

    int allocate() {
        if (free_blocks_.empty()) {
            throw std::runtime_error("paged KV cache OOM: no free blocks");
        }
        const int block = free_blocks_.back();
        free_blocks_.pop_back();
        free_set_.erase(block);
        used_set_.insert(block);
        return block;
    }

    std::vector<int> allocate_many(int count) {
        if (count < 0) {
            throw std::invalid_argument("count must be non-negative");
        }
        if (count > available()) {
            throw std::runtime_error("paged KV cache OOM: insufficient free blocks");
        }

        std::vector<int> blocks;
        blocks.reserve(count);
        for (int i = 0; i < count; ++i) {
            blocks.push_back(allocate());
        }
        return blocks;
    }

    void release(int block) {
        if (block < 0 || block >= total_blocks_) {
            throw std::invalid_argument("invalid physical block id");
        }
        if (free_set_.count(block) != 0) {
            throw std::runtime_error("double free detected");
        }
        if (used_set_.count(block) == 0) {
            throw std::runtime_error("block was not allocated by this pool");
        }

        used_set_.erase(block);
        free_set_.insert(block);
        free_blocks_.push_back(block);
    }

    void release_many(const std::vector<int>& blocks) {
        for (int block : blocks) {
            release(block);
        }
    }

    int available() const {
        return static_cast<int>(free_blocks_.size());
    }

    int used() const {
        return static_cast<int>(used_set_.size());
    }

    int total_blocks() const {
        return total_blocks_;
    }

private:
    int total_blocks_ = 0;
    std::vector<int> free_blocks_;
    std::unordered_set<int> free_set_;
    std::unordered_set<int> used_set_;
};
