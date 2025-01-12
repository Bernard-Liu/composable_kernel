/*
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/torch.h>

#include <hip/hip_runtime.h>

#include "ck_tile/host/hip_check_error.hpp"

#include "paged_attention.hpp"

void paged_attention(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& tmp_out,     // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,       // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor& value_cache, // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& context_lens, // [num_seqs]
    int64_t block_size,
    int64_t max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    double k_scale,
    double v_scale,
    const c10::optional<torch::Tensor>& fp8_out_scale,
    int64_t partition_size)
{

    native::paged_attention_traits traits;

    traits.q_type         = (query.dtype() == at::ScalarType::Half ? native::ScalarType::Half
                                                                   : native::ScalarType::BFloat16);
    traits.kv_cache_dtype = kv_cache_dtype;

    native::paged_attention_args args;

    args.head_size = query.size(2);

    args.num_seqs               = query.size(0);
    args.num_heads              = query.size(1);
    args.head_size              = query.size(2);
    args.max_num_blocks_per_seq = block_tables.size(1);
    args.q_stride               = query.stride(0);
    args.kv_block_stride        = key_cache.stride(0);
    args.kv_head_stride         = key_cache.stride(1);

    // NOTE: alibi_slopes is optional.
    args.alibi_slopes_ptr =
        alibi_slopes ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr()) : nullptr;

    args.exp_sums_ptr     = reinterpret_cast<float*>(exp_sums.data_ptr());
    args.max_logits_ptr   = reinterpret_cast<float*>(max_logits.data_ptr());
    args.tmp_out_ptr      = tmp_out.data_ptr();
    args.query_ptr        = query.data_ptr();
    args.key_cache_ptr    = key_cache.data_ptr();
    args.value_cache_ptr  = value_cache.data_ptr();
    args.block_tables_ptr = block_tables.data_ptr<int>();
    args.context_lens_ptr = context_lens.data_ptr<int>();

    // NOTE: fp8_out_scale is optional.
    args.fp8_out_scale_ptr =
        fp8_out_scale ? reinterpret_cast<const float*>(fp8_out_scale.value().data_ptr()) : nullptr;
    args.out_ptr = out.data_ptr();

    args.block_size = block_size;

    args.max_context_len = max_context_len;
    args.num_kv_heads    = num_kv_heads;
    args.partition_size  = partition_size;
    args.scale           = scale;
    args.k_scale         = k_scale;
    args.v_scale         = v_scale;

    hipStream_t stream = nullptr;
    HIP_CHECK_ERROR(hipStreamCreate(&stream));

    native::paged_attention(traits, args, stream);

    HIP_CHECK_ERROR(hipStreamDestroy(stream));
}
