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

#include "kernel/attention_kernel.hpp"

#define LAUNCH_CUSTOM_ATTENTION(GQA_RATIO)                                    \
  paged_attention_ll4mi_QKV_kernel<T, KVT, KV_DTYPE, OUTT, BLOCK_SIZE,        \
                                   HEAD_SIZE, NTHR, GQA_RATIO>                \
      <<<grid, block, 0, stream>>>(                                           \
          query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale,     \
          block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,         \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,        \
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, out_ptr, max_ctx_blocks, \
          k_scale, v_scale, fp8_out_scale_ptr);

#define LAUNCH_CUSTOM_REDUCTION(NPAR_LOOPS)                          \
  paged_attention_ll4mi_reduce_kernel<T, OUTT, HEAD_SIZE, HEAD_SIZE, \
                                      PARTITION_SIZE, NPAR_LOOPS>    \
      <<<reduce_grid, reduce_block, 0, stream>>>(                    \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr,        \
          context_lens_ptr, max_num_partitions, fp8_out_scale_ptr);

template <typename T, typename KVT, vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE, int HEAD_SIZE, typename OUTT, int PARTITION_SIZE>
void paged_attention_custom_launcher(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, const int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& context_lens,
    int max_context_len, const c10::optional<torch::Tensor>& alibi_slopes,
    float k_scale, float v_scale,
    const c10::optional<torch::Tensor>& fp8_out_scale) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  KVT* key_cache_ptr = reinterpret_cast<KVT*>(key_cache.data_ptr());
  KVT* value_cache_ptr = reinterpret_cast<KVT*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  // NOTE: fp8_out_scale is optional.
  const float* fp8_out_scale_ptr =
      fp8_out_scale
          ? reinterpret_cast<const float*>(fp8_out_scale.value().data_ptr())
          : nullptr;
  OUTT* out_ptr = reinterpret_cast<OUTT*>(out.data_ptr());

  const int max_ctx_blocks = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE);
  const int max_num_partitions =
      DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  const int gqa_ratio = num_heads / num_kv_heads;
  assert(num_heads % num_kv_heads == 0);
  assert(head_size == HEAD_SIZE);

  constexpr int NTHR = PARTITION_SIZE;
  dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
  dim3 block(NTHR);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (gqa_ratio) {
    case 1:
      LAUNCH_CUSTOM_ATTENTION(1);
      break;
    case 2:
      LAUNCH_CUSTOM_ATTENTION(2);
      break;
    case 3:
      LAUNCH_CUSTOM_ATTENTION(3);
      break;
    case 4:
      LAUNCH_CUSTOM_ATTENTION(4);
      break;
    case 5:
      LAUNCH_CUSTOM_ATTENTION(5);
      break;
    case 6:
      LAUNCH_CUSTOM_ATTENTION(6);
      break;
    case 7:
      LAUNCH_CUSTOM_ATTENTION(7);
      break;
    case 8:
      LAUNCH_CUSTOM_ATTENTION(8);
      break;
    case 9:
      LAUNCH_CUSTOM_ATTENTION(9);
      break;
    case 10:
      LAUNCH_CUSTOM_ATTENTION(10);
      break;
    case 11:
      LAUNCH_CUSTOM_ATTENTION(11);
      break;
    case 12:
      LAUNCH_CUSTOM_ATTENTION(12);
      break;
    case 13:
      LAUNCH_CUSTOM_ATTENTION(13);
      break;
    case 14:
      LAUNCH_CUSTOM_ATTENTION(14);
      break;
    case 15:
      LAUNCH_CUSTOM_ATTENTION(15);
      break;
    case 16:
      LAUNCH_CUSTOM_ATTENTION(16);
      break;
    default:
      TORCH_CHECK(false, "Unsupported gqa ratio: ", gqa_ratio);
      break;
  }

  // reduction kernel is only required if max_context_len > partition size,
  // otherwise main kernel writes directly to final output
  //  note there are cases with graphing where max_context_len is the max
  //  supported by graphing, not the actual max among all the sequences: in that
  //  case reduction kernel will still run but return immediately
  if (max_context_len > PARTITION_SIZE) {
    dim3 reduce_grid(num_heads, num_seqs);
    dim3 reduce_block(head_size);
    const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, WARP_SIZE);
    // support upto 8*64*256=128K context length
    switch (npar_loops) {
      case 1:
        LAUNCH_CUSTOM_REDUCTION(1);
        break;
      case 2:
        LAUNCH_CUSTOM_REDUCTION(2);
        break;
      case 3:
        LAUNCH_CUSTOM_REDUCTION(3);
        break;
      case 4:
        LAUNCH_CUSTOM_REDUCTION(4);
        break;
      case 5:
        LAUNCH_CUSTOM_REDUCTION(5);
        break;
      case 6:
        LAUNCH_CUSTOM_REDUCTION(6);
        break;
      case 7:
        LAUNCH_CUSTOM_REDUCTION(7);
        break;
      case 8:
        LAUNCH_CUSTOM_REDUCTION(8);
        break;
      default:
        TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops);
        break;
    }
  }
}

#define CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT,      \
                             PSIZE)                                            \
  paged_attention_custom_launcher<T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, \
                                  PSIZE>(                                      \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,       \
      num_kv_heads, scale, block_tables, context_lens, max_context_len,        \
      alibi_slopes, k_scale, v_scale, fp8_out_scale);

#define CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,     \
                                   OUTT)                                      \
  switch (partition_size) {                                                   \
    case 256:                                                                 \
      CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, 256); \
      break;                                                                  \
    case 512:                                                                 \
      CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, 512); \
      break;                                                                  \
    default:                                                                  \
      TORCH_CHECK(false, "Unsupported partition size: ", partition_size);     \
      break;                                                                  \
  }

#if defined(__HIPCC__) && defined(__gfx90a__)
  #define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE)   \
    if (fp8_out_scale) {                                                    \
      TORCH_CHECK(false, "fp8 out scale unsupported for gfx90a");           \
    } else {                                                                \
      CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T); \
    }
#else
  #define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE)   \
    if (fp8_out_scale) {                                                    \
      CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,     \
                                 uint8_t);                                  \
    } else {                                                                \
      CALL_CUSTOM_LAUNCHER_PSIZE(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T); \
    }
#endif
#define CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, HEAD_SIZE)     \
  switch (block_size) {                                           \
    case 16:                                                      \
      CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 16, HEAD_SIZE);  \
      break;                                                      \
    case 32:                                                      \
      CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 32, HEAD_SIZE);  \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

#define CALL_CUSTOM_LAUNCHER_BLK_HEAD(T, KVT, KV_DTYPE)         \
  switch (head_size) {                                          \
    case 64:                                                    \
      CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 64);           \
      break;                                                    \
    case 128:                                                   \
      CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 128);          \
      break;                                                    \
    default:                                                    \
      TORCH_CHECK(false, "Unsupported head size: ", head_size); \
      break;                                                    \
  }

void paged_attention(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor&
        tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& context_lens,  // [num_seqs]
    int64_t block_size, int64_t max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double k_scale, double v_scale,
    const c10::optional<torch::Tensor>& fp8_out_scale, int64_t partition_size) {
  const int head_size = query.size(2);
  if (kv_cache_dtype == "auto") {
    if (query.dtype() == at::ScalarType::Half) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, _Float16,
                                    vllm::Fp8KVCacheDataType::kAuto);
    } else if (query.dtype() == at::ScalarType::BFloat16) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, __hip_bfloat16,
                                    vllm::Fp8KVCacheDataType::kAuto);
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    if (query.dtype() == at::ScalarType::Half) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, uint8_t,
                                    vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (query.dtype() == at::ScalarType::BFloat16) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, uint8_t,
                                    vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else {
    TORCH_CHECK(false, "Unsupported KV cache dtype: ", kv_cache_dtype);
  }
}