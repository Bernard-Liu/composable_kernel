// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

#include <type_traits>

// mla get metadata

namespace ck_tile {
template <typename Problem_>

struct GetMetadataHostArgs
{
	const void* seqlens_k_ptr;
	void* tile_scheduler_metadata_ptr;
	void* num_splits_ptr;
	int batch_size;
	int block_size_n;
	int fixed_overhead_num_blocks;
	int num_sm_parts;
};

struct MlaGetMetadataKernel
{
	using Problem = ck_tile::remove_cvref_t<Problem>;
	static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;
	static constexpr ck_tile::index_t kMaxBatchSize = Problem::kMaxBatchSize;
	static constexpr ck_tile::index_t kMetadataSize = Problem::kMetadataSize;

	using IndexType = ck_tile::remove_cvref_t<Problem::IndexType>;
	
	struct GetMetadataKargs
	{
		const void* seqlens_k_ptr;
		void* tile_scheduler_metadata_ptr;
		void* num_splits_ptr;
		int batch_size;
		int block_size_n;
		int fixed_overhead_num_blocks;
		int num_sm_parts;
	};

	struct TileSchedulerMetadata
	{
		index_t start_batch_id;
		index_t start_block_id;
		index_t end_batch_id;
		index_t end_block_id;
		index_t n_split_idx;
	};

	CK_TILE_HOST static constexpr auto GridSize(GetMetadataHostArgs hargs)
	{
	
	}

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return 2 * kMaxBatchSize * sizeof(IndexType);
    }

	CK_TILE_HOST_DEVICE static constexpr auto BlockSize() { return Problem::kBlockSize; }


	CK_TILE_DEVICE void
	get_mla_metadata(const IndexType* __restrict__ seqlens_k_ptr,
					 IndexType* __restrict__ tile_scheduler_metadata_ptr,
					 IndexType* __restrict__ num_splits_ptr,
					 index_t batch_size,
					 index_t block_size_n,
					 index_t fixed_overhead_num_blocks,
					 index_t num_sm_parts,
					 void *smem)
	{
        IndexType* num_blocks_shared = reinterpret_cast<IndexType*>(smem);
        IndexType* num_splits_shared = reinterpret_cast<IndexType*>(smem + kMaxBatchSize * sizeof(IndexType));

        const index_t tid = static_cast<index_t>(threadIdx.x);
		int total_num_blocks = 0;

		for(int i = tid; i < batch_size; i += 64)
		{
			int num_blocks = ck_tile::integer_divide_ceil(seqlens_k_ptr[i], block_size_n);
			total_num_blocks += num_blocks + fixed_overhead_num_blocks;
			num_blocks_shared[i] = num_blocks;
		}

		{
			auto reduce_op = [&](auto x_, auto y_) { return x_ + y_; };
			int v_remote_tmp = __builtin_amdgcn_ds_bpermute(((__lane_id() & 0x30) - 17) << 2, __builtin_bit_cast(int, total_num_blocks));
			v_remote_tmp = __lane_id() >= 32 ? v_remote_tmp : 0;
			total_num_blocks = reduce_op(total_num_blocks, __builtin_bit_cast(data_t, v_remote_tmp));
		}

		if (tid == 0)
		{
			int payload = ck_tile::integer_divide_ceil(total_num_blocks, kargs.num_sm_parts) + args.fixed_overhead_num_blocks;
			int now_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
			for(int part_id = 0; part_id < num_sm_parts; ++part_id)
			{
				int tile_scheduler_metadata[kMetadataSize];
				tile_scheduler_metadata[0] = now_idx;
				tile_scheduler_metadata[1] = now_block;
				tile_scheduler_metadata[4] = now_n_split_idx;
				int remain_payload = payload;
				while(now_idx < batch_size)
				{
					int num_blocks = num_blocks_shared[now_idx];
					int now_remain_blocks = num_blocks - now_block;
					if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks)
					{
						cum_num_splits += now_n_split_idx + 1;
						num_splits_shared[now_idx + 1] = cum_num_splits;
						remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
						++now_idx;
						now_blocK = 0;
						now_n_split_idx = 0;
					}
					else
					{
						if(remain_payload - fixed_overhead_num_blocks > 0)
						{
							now_block += remain_payload - fixed_overhead_num_blocks;
							++now_n_split_idx;
							remain_payload = 0;
						}
						break;
					}
				}
				tile_scheduler_metadata[2] = now_block > 0 ? now_idx : now_idx - 1;
				tile_scheduler_metadata[3] = now_block > 0 ? now_block * block_size_n : seqlens_k_ptr[now_idx - 1];
				static_for<0, kMetadataSize, 1>{}([&](auto idx) {
					tile_scheduler_metadata_ptr[part_id * TileSchedulerMetaDataSize + idx] = tile_scheduler_metadata[idx];
				});
			}
			static_assert(now_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
		}

		for(int i = tid; i <= batch_size; i += 64)
			num_splits_ptr[i] = num_splits_shared[i];
	}

	CK_TILE_DEVICE void operator()(GetMetadataKargs kargs) const
	{
        __shared__ char smem[GetSmemSize()];
	}
};
}
