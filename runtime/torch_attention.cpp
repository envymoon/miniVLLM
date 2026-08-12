#include <torch/extension.h>

void append_kv_cache_cuda(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping,
    torch::Tensor active_mask);

void flash_prefill_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor positions,
    torch::Tensor request_indices,
    torch::Tensor block_tables,
    torch::Tensor is_decode,
    torch::Tensor active_mask,
    int64_t block_size,
    torch::Tensor output);

void paged_decode_cuda(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor query_start_loc,
    torch::Tensor seq_lens,
    torch::Tensor block_tables,
    torch::Tensor is_decode,
    torch::Tensor active_mask,
    int64_t block_size,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("append_kv_", &append_kv_cache_cuda);
    module.def("flash_prefill_out", &flash_prefill_cuda);
    module.def("paged_decode_out", &paged_decode_cuda);
}
