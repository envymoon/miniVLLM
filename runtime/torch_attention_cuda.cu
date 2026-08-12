#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void append_kv_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int* __restrict__ slot_mapping,
    const bool* __restrict__ active_mask,
    int total_tokens,
    int kv_heads,
    int head_dim) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements_per_token = kv_heads * head_dim;
    const int total_elements = total_tokens * elements_per_token;
    if (linear >= total_elements) {
        return;
    }
    const int token = linear / elements_per_token;
    if (!active_mask[token]) {
        return;
    }
    const int within = linear % elements_per_token;
    const int slot = slot_mapping[token];
    const int cache_index = slot * elements_per_token + within;
    key_cache[cache_index] = key[linear];
    value_cache[cache_index] = value[linear];
}

template <typename scalar_t>
__device__ void online_paged_attention(
    const scalar_t* query,
    const scalar_t* key_cache,
    const scalar_t* value_cache,
    const int* block_table,
    int context_len,
    int block_size,
    int query_heads,
    int kv_heads,
    int head_dim,
    int query_index,
    int query_head,
    scalar_t* output,
    float* reduction,
    float* stats) {
    const int groups = query_heads / kv_heads;
    const int kv_head = query_head / groups;
    const int dim = threadIdx.x;
    float accumulator = 0.0f;
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (int token = 0; token < context_len; ++token) {
        float partial = 0.0f;
        if (dim < head_dim) {
            const int physical_block = block_table[token / block_size];
            const int slot = physical_block * block_size + token % block_size;
            const int query_offset =
                (query_index * query_heads + query_head) * head_dim + dim;
            const int key_offset =
                (slot * kv_heads + kv_head) * head_dim + dim;
            partial = static_cast<float>(query[query_offset]) *
                      static_cast<float>(key_cache[key_offset]);
        }
        reduction[dim] = partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (dim < stride) {
                reduction[dim] += reduction[dim + stride];
            }
            __syncthreads();
        }
        if (dim == 0) {
            const float score = reduction[0] * scale;
            const float new_max = fmaxf(row_max, score);
            stats[0] = row_max == -INFINITY ? 0.0f : expf(row_max - new_max);
            stats[1] = expf(score - new_max);
            stats[2] = new_max;
            stats[3] = stats[0] * row_sum + stats[1];
        }
        __syncthreads();
        if (dim < head_dim) {
            const int physical_block = block_table[token / block_size];
            const int slot = physical_block * block_size + token % block_size;
            const int value_offset =
                (slot * kv_heads + kv_head) * head_dim + dim;
            accumulator = stats[0] * accumulator +
                          stats[1] * static_cast<float>(value_cache[value_offset]);
        }
        row_max = stats[2];
        row_sum = stats[3];
        __syncthreads();
    }
    if (dim < head_dim) {
        const int output_offset =
            (query_index * query_heads + query_head) * head_dim + dim;
        output[output_offset] = static_cast<scalar_t>(accumulator / row_sum);
    }
}

template <typename scalar_t>
__global__ void flash_prefill_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const int* __restrict__ positions,
    const int* __restrict__ request_indices,
    const int* __restrict__ block_tables,
    const bool* __restrict__ is_decode,
    const bool* __restrict__ active_mask,
    int total_tokens,
    int query_heads,
    int kv_heads,
    int head_dim,
    int block_size,
    int max_blocks,
    scalar_t* __restrict__ output) {
    const int token = blockIdx.x;
    const int query_head = blockIdx.y;
    if (token >= total_tokens || query_head >= query_heads ||
        !active_mask[token]) {
        return;
    }
    const int request = request_indices[token];
    if (is_decode[request]) {
        return;
    }
    extern __shared__ float shared[];
    float* reduction = shared;
    float* stats = shared + blockDim.x;
    online_paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables + request * max_blocks,
        positions[token] + 1,
        block_size,
        query_heads,
        kv_heads,
        head_dim,
        token,
        query_head,
        output,
        reduction,
        stats);
}

template <typename scalar_t>
__global__ void paged_decode_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ seq_lens,
    const int* __restrict__ block_tables,
    const bool* __restrict__ is_decode,
    const bool* __restrict__ active_mask,
    int num_requests,
    int query_heads,
    int kv_heads,
    int head_dim,
    int block_size,
    int max_blocks,
    scalar_t* __restrict__ output) {
    const int request = blockIdx.x;
    const int query_head = blockIdx.y;
    if (request >= num_requests || query_head >= query_heads ||
        !is_decode[request]) {
        return;
    }
    const int query_index = query_start_loc[request];
    if (!active_mask[query_index]) {
        return;
    }
    extern __shared__ float shared[];
    float* reduction = shared;
    float* stats = shared + blockDim.x;
    online_paged_attention(
        query,
        key_cache,
        value_cache,
        block_tables + request * max_blocks,
        seq_lens[request],
        block_size,
        query_heads,
        kv_heads,
        head_dim,
        query_index,
        query_head,
        output,
        reduction,
        stats);
}

void check_common(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& output) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda(), "cache must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous(), "key cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(), "value cache must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(query.scalar_type() == key_cache.scalar_type(), "dtype mismatch");
    TORCH_CHECK(query.scalar_type() == value_cache.scalar_type(), "dtype mismatch");
    TORCH_CHECK(query.scalar_type() == output.scalar_type(), "dtype mismatch");
    TORCH_CHECK(query.device() == key_cache.device(), "device mismatch");
    TORCH_CHECK(query.device() == value_cache.device(), "device mismatch");
    TORCH_CHECK(query.device() == output.device(), "device mismatch");
    TORCH_CHECK(query.dim() == 3, "query must have shape [tokens, heads, dim]");
    TORCH_CHECK(key_cache.dim() == 3, "key cache must have shape [slots, heads, dim]");
    TORCH_CHECK(query.size(2) == key_cache.size(2), "head_dim mismatch");
    TORCH_CHECK(query.size(2) <= kThreads, "head_dim must be <= 256");
    TORCH_CHECK(query.size(1) % key_cache.size(1) == 0, "invalid GQA ratio");
    TORCH_CHECK(query.sizes() == output.sizes(), "output shape mismatch");
}

void check_metadata(
    const torch::Tensor& tensor,
    at::ScalarType dtype,
    const char* name,
    const torch::Device& device) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
    TORCH_CHECK(tensor.device() == device, name, " is on the wrong device");
}

}  // namespace

void append_kv_cache_cuda(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor slot_mapping,
    torch::Tensor active_mask) {
    TORCH_CHECK(key.is_cuda() && value.is_cuda(), "K/V must be CUDA");
    TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda(), "cache must be CUDA");
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous(), "K/V must be contiguous");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping must be int32");
    TORCH_CHECK(active_mask.scalar_type() == at::kBool, "active_mask must be bool");
    TORCH_CHECK(key.sizes() == value.sizes(), "K/V shape mismatch");
    TORCH_CHECK(key.scalar_type() == key_cache.scalar_type(), "K/cache dtype mismatch");
    TORCH_CHECK(value.scalar_type() == value_cache.scalar_type(), "V/cache dtype mismatch");
    TORCH_CHECK(key.device() == key_cache.device(), "K/cache device mismatch");
    TORCH_CHECK(value.device() == value_cache.device(), "V/cache device mismatch");
    check_metadata(slot_mapping, at::kInt, "slot_mapping", key.device());
    check_metadata(active_mask, at::kBool, "active_mask", key.device());
    const c10::cuda::CUDAGuard guard(key.device());
    const int total_tokens = static_cast<int>(key.size(0));
    const int kv_heads = static_cast<int>(key.size(1));
    const int head_dim = static_cast<int>(key.size(2));
    const int elements = total_tokens * kv_heads * head_dim;
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        key.scalar_type(),
        "append_kv_cache_cuda",
        [&] {
            append_kv_kernel<scalar_t><<<(elements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                key_cache.data_ptr<scalar_t>(),
                value_cache.data_ptr<scalar_t>(),
                slot_mapping.data_ptr<int>(),
                active_mask.data_ptr<bool>(),
                total_tokens,
                kv_heads,
                head_dim);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    torch::Tensor output) {
    check_common(query, key_cache, value_cache, output);
    check_metadata(positions, at::kInt, "positions", query.device());
    check_metadata(
        request_indices, at::kInt, "request_indices", query.device());
    check_metadata(block_tables, at::kInt, "block_tables", query.device());
    check_metadata(is_decode, at::kBool, "is_decode", query.device());
    check_metadata(active_mask, at::kBool, "active_mask", query.device());
    const c10::cuda::CUDAGuard guard(query.device());
    const int total_tokens = static_cast<int>(query.size(0));
    const int query_heads = static_cast<int>(query.size(1));
    const int kv_heads = static_cast<int>(key_cache.size(1));
    const int head_dim = static_cast<int>(query.size(2));
    const int max_blocks = static_cast<int>(block_tables.size(1));
    const dim3 grid(total_tokens, query_heads, 1);
    const size_t shared_bytes = (kThreads + 4) * sizeof(float);
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        query.scalar_type(),
        "flash_prefill_cuda",
        [&] {
            flash_prefill_kernel<scalar_t><<<grid, kThreads, shared_bytes, stream>>>(
                query.data_ptr<scalar_t>(),
                key_cache.data_ptr<scalar_t>(),
                value_cache.data_ptr<scalar_t>(),
                positions.data_ptr<int>(),
                request_indices.data_ptr<int>(),
                block_tables.data_ptr<int>(),
                is_decode.data_ptr<bool>(),
                active_mask.data_ptr<bool>(),
                total_tokens,
                query_heads,
                kv_heads,
                head_dim,
                static_cast<int>(block_size),
                max_blocks,
                output.data_ptr<scalar_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    torch::Tensor output) {
    check_common(query, key_cache, value_cache, output);
    check_metadata(
        query_start_loc, at::kInt, "query_start_loc", query.device());
    check_metadata(seq_lens, at::kInt, "seq_lens", query.device());
    check_metadata(block_tables, at::kInt, "block_tables", query.device());
    check_metadata(is_decode, at::kBool, "is_decode", query.device());
    check_metadata(active_mask, at::kBool, "active_mask", query.device());
    const c10::cuda::CUDAGuard guard(query.device());
    const int num_requests = static_cast<int>(seq_lens.size(0));
    const int query_heads = static_cast<int>(query.size(1));
    const int kv_heads = static_cast<int>(key_cache.size(1));
    const int head_dim = static_cast<int>(query.size(2));
    const int max_blocks = static_cast<int>(block_tables.size(1));
    const dim3 grid(num_requests, query_heads, 1);
    const size_t shared_bytes = (kThreads + 4) * sizeof(float);
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        query.scalar_type(),
        "paged_decode_cuda",
        [&] {
            paged_decode_kernel<scalar_t><<<grid, kThreads, shared_bytes, stream>>>(
                query.data_ptr<scalar_t>(),
                key_cache.data_ptr<scalar_t>(),
                value_cache.data_ptr<scalar_t>(),
                query_start_loc.data_ptr<int>(),
                seq_lens.data_ptr<int>(),
                block_tables.data_ptr<int>(),
                is_decode.data_ptr<bool>(),
                active_mask.data_ptr<bool>(),
                num_requests,
                query_heads,
                kv_heads,
                head_dim,
                static_cast<int>(block_size),
                max_blocks,
                output.data_ptr<scalar_t>());
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
