// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

#include <string>
#include <type_traits>
#include <variant>

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_qk] @ K[seqlen_k, hdim_qk]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// mla don't have alibi bias
// P[seqlen_q, seqlen_k] = Softmax(S[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] @ V^T[hdim_v, seqlen_k]

namespace ck_tile {
template <typename FmlaPipeline_, typename FmlaEpiloguePipeline_>
struct FmlaFwdSplitKVKernel
{
	using FmlaPipeline = ck_tile::remove_cvref_t<FmlaPipeline_>;
	using FmlaEpiloguePipeline = ck_tile::remove_cvref_t<FmlaEpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmlaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmlaPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);
    static constexpr ck_tile::index_t kBlockPerCuInput = FmlaPipeline::Problem::kBlockPerCu;
	static constexpr ck_tile::index_t kNumThreadsPerWarpGroup = FmlaPipeline::Problem::kNumThreadsPerWarpGroup;

    using QDataType    = ck_tile::remove_cvref_t<typename FmlaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmlaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmlaPipeline::VDataType>;

    using LSEDataType  = ck_tile::remove_cvref_t<typename FmlaPipeline::LSEDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename FmlaPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmlaPipeline::SaccDataType>;

    static constexpr bool kPadSeqLenQ       = FmlaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = FmlaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = FmlaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = FmlaPipeline::kPadHeadDimV;
    static constexpr bool kStoreLSE         = FmlaPipeline::kStoreLSE;
    static constexpr bool kDoFp8StaticQuant = FmlaPipeline::Problem::kDoFp8StaticQuant;
    static constexpr bool kIsPagedKV        = FmlaPipeline::Problem::kIsPagedKV;
    using FmlaMask                 = ck_tile::remove_cvref_t<typename FmlaPipeline::FmlaMask>;
    static constexpr bool kHasMask = FmlaMask::IsMasking;

    struct FmlaFwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* o_acc_ptr;

        ck_tile::index_t batch,
        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        ck_tile::index_t nhead_ratio_qk;
        float scale_s;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_o_acc;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_o_acc;

		const void* tile_scheduler_metadata_ptr;
		const void* num_splits_ptr;
		int num_sm_parts;
    };

    struct FmlaFwdFp8StaticQuantKargs
    {
        float scale_p;
    };

    struct CommonPageBlockTableKargs
    {
        const int32_t* block_table_ptr;
        ck_tile::index_t batch_stride_block_table;
        ck_tile::index_t page_block_size;
    };

    struct FmlaFwdMaskKargs
    {
        // ck_tile::index_t window_size_left, window_size_right;
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    struct FmlaFwdCommonLSEKargs
    {
        void* lse_acc_ptr                 = nullptr;
        ck_tile::index_t nhead_stride_lse_acc = 0;
        ck_tile::index_t batch_stride_lse_acc = 0;
    };

    struct FmlaFwdBatchModeKargs
        : FmlaFwdCommonKargs,
          std::conditional_t<kHasMask, FmlaFwdMaskKargs, FmlaFwdEmptyKargs<0>>,
          std::conditional_t<kStoreLSE, FmlaFwdCommonLSEKargs, FmlaFwdEmptyKargs<1>>,
          std::conditional_t<kDoFp8StaticQuant, FmlaFwdFp8StaticQuantKargs, FmlaFwdEmptyKargs<2>>>
          std::conditional_t<kIsPagedKV, CommonPageBlockTableKargs, CacheBatchIdxKargs>
    {
        const void* seqlen_k_ptr;
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_o_acc;
    };

    CK_TILE_HOST static constexpr
    MakeKargsImpl(const void* q_ptr,
                  const void* k_ptr,
                  const void* v_ptr,
                  void* lse_acc_ptr,
                  void* o_acc_ptr,
                  ck_tile::index_t batch,
                  ck_tile::index_t seqlen_q,
                  ck_tile::index_t seqlen_k,
				  const void* seqlen_k_ptr,  // only used for (paged-) kvcache
                  ck_tile::index_t hdim_q,
                  ck_tile::index_t hdim_v,
                  ck_tile::index_t num_head_q,
                  ck_tile::index_t nhead_ratio_qk,
				  const void* tile_scheduler_metadata_ptr,
				  const void* num_splits_ptr,
				  const void* block_table_ptr,
				  ck_tile::index_t batch_stride_block_table,
				  ck_tile::index_t page_block_size,
				  // const void* cache_batch_idx,
                  float scale_s,
                  float scale_p,
                  ck_tile::index_t stride_q,
                  ck_tile::index_t stride_k,
                  ck_tile::index_t stride_v,
                  ck_tile::index_t stride_o_acc,
                  ck_tile::index_t nhead_stride_q,
                  ck_tile::index_t nhead_stride_k,
                  ck_tile::index_t nhead_stride_v,
                  ck_tile::index_t nhead_stride_lse_acc,
                  ck_tile::index_t nhead_stride_o_acc,
                  ck_tile::index_t batch_stride_q,
                  ck_tile::index_t batch_stride_k,
                  ck_tile::index_t batch_stride_v,
                  ck_tile::index_t batch_stride_lse_acc,
                  ck_tile::index_t batch_stride_o_acc,
                  ck_tile::index_t window_size_left,
                  ck_tile::index_t window_size_right,
                  ck_tile::index_t mask_type)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_acc_ptr,
                     batch,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o_acc,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o_acc}, // args for common karg
                    {},                   // placeholder for mask
                    {},                   // placeholder for lse
                    {},                   // placeholder for fp8_static_quant args
                    {},                   // placeholder for paged-block table or cache_batch_idx
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr),
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_o};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_acc_ptr      = lse_acc_ptr;
            kargs.nhead_stride_lse_acc = nhead_stride_lse_acc;
            kargs.batch_stride_lse_acc = batch_stride_lse_acc;
        }
        if constexpr(kDoFp8StaticQuant)
        {
            kargs.scale_p = scale_p;
        }
        if constexpr(kIsPagedKV)
        {
            kargs.block_table_ptr          = reinterpret_cast<const int32_t*>(block_table_ptr);
            kargs.batch_stride_block_table = batch_stride_block_table;
            kargs.page_block_size          = page_block_size;
        }

        return kargs;
    }

    CK_TILE_HOST static constexpr
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              void* lse_acc_ptr,
              void* o_acc_ptr,
              ck_tile::index_t batch,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
			  const void* seqlen_k_ptr,  // only used for (paged-) kvcache
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
			  const void* tile_scheduler_metadata_ptr,
			  const void* num_splits_ptr,
			  const void* block_table_ptr,
			  ck_tile::index_t batch_stride_block_table,
			  ck_tile::index_t page_block_size,
			  // const void* cache_batch_idx,
              float scale_s,
              float scale_p,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_o_acc,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_lse_acc,
              ck_tile::index_t nhead_stride_o_acc,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_lse_acc,
              ck_tile::index_t batch_stride_o_acc,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type)
    {
        return MakeKargsImpl(
            q_ptr,
            k_ptr,
            v_ptr,
            lse_acc_ptr,
            o_acc_ptr,
            batch,
            seqlen_q,
            seqlen_k,
			seqlen_k_ptr,
            hdim_q,
            hdim_v,
            num_head_q,
			nhead_ratio_qk,
            tile_scheduler_metadata_ptr,
            num_splits_ptr,
            block_table_ptr,
            batch_stride_block_table,
            page_block_size,
            scale_s,
            scale_p,
            stride_q,
            stride_k,
            stride_v,
            stride_o_acc,
            nhead_stride_q,
            nhead_stride_k,
            nhead_stride_v,
            nhead_stride_lse_acc,
            nhead_stride_o_acc,
            batch_stride_q,
            batch_stride_k,
            batch_stride_v,
            batch_stride_lse_acc,
            batch_stride_o_acc,
            window_size_left,
            window_size_right,
            mask_type,
    }

	CK_TIlE_HOST static constexpr auto GridSize(ck_tile::index_t nhead_q_,
												ck_tile::index_t seqlen_q_,
												ck_tile::index_t num_sm_parts_)
	{
		// index_t num_sm_parts = sm_count / num_heads_k / ck_tile::integer_divide_ceil(seqlen_q_ * nhead_ratio_qk_, FmlaPipeline::kM0);
		return dim3(ck_tile::integer_divide_ceil(seqlen_q_, FmlaPipeline::kM0),
					nhead_q_,
					num_sm_parts);
    }

	template<int kHeadDimV>
    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
		const index_t i_block_m = blockIdx.x;
		const index_t i_nhead = blockIdx.y;
		const index_t i_partition = blockIdx.z;

		return ck_tile::make_tuple(i_block_m, i_nhead, i_partition);
	}

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FmlaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_HOST static constexpr auto GetWaveGroupId() { return threadIdx.x / kNumThreadsPerWarpGroup; }

	// CK_TILE_DEVICE void ComputeAttn1RowBlockSplitKVMLA(Kargs kargs,
 //                                                       const int batch_id,
 //                                                       const int i_block_m,
 //                                                       const int i_nhead,
 //                                                       const int n_split_idx,
 //                                                       const int seqlen_k,
 //                                                       const int n_block_min,
 //                                                       const int n_block_max,
 //                                                       const bool NoSplit
 //                                                       void* smem_ptr)
	// {
	// }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_block_m, i_nhead, i_partition] = GetTileIndex(kargs);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_block * FmlaPipeline::kM0);
		IndexDataType* tile_scheduler_metadata_ptr = reinterpret_cast<IndexDataType*>(kargs.tile_scheduler_metadata_ptr) + i_partition * TileSchedulerMetaDataSize;

		IndexDataType begin_idx         = tile_scheduler_metadata_ptr[0];
		IndexDataType begin_seqlen      = tile_scheduler_metadata_ptr[1];
		IndexDataType end_idx           = tile_scheduler_metadata_ptr[2];
		IndexDataType end_seqlen        = tile_scheduler_metadata_ptr[3];
		if(begin_idx >= kargs.batch)
			return;
		IndexDataType begin_n_split_idx = tile_scheduler_metadata_ptr[4];


        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&] {
            const auto q_dram_naive = [&] {
				return make_naive_tensor_view<address_space_enum::global>(
					nullptr,
					make_tuple(kargs.seqlen_q, kargs.hdim_q),
					make_tuple(kargs.stride_q, 1),
					number<FmlaPipeline::kAlignmentQ>{},
					number<1>{});
			}();
            if constexpr(FmlaPipeline::kQLoadOnce)
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmlaPipeline::kM0>{}, number<FmlaPipeline::kSubQKHeaddim>{}),
                    sequence<false, kPadHeadDimQ>{});
            }
            else
            {
                return pad_tensor_view(
                    q_dram_naive,
                    make_tuple(number<FmlaPipeline::kM0>{}, number<FmlaPipeline::kK0>{}),
                    sequence<false, kPadHeadDimQ>{});
            }
        }();
		
        const auto make_k_dram = [&](const KDataType* data, index_t height) {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                data, // will update this pointer if using paged-kvcache
                make_tuple(height, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<FmlaPipeline::kAlignmentK>{},
                number<1>{});

            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<FmlaPipeline::kN0>{}, number<FmlaPipeline::kK0>{}),
                sequence<false, kPadHeadDimQ>{});
        };
        const auto k_dram = [&]() {
            if constexpr(kIsPagedKV)
            {
                return make_k_dram(nullptr, kargs.page_block_size);
            }
        }();

		// v is shared with k, so the layout of v must be equal to k's
        const auto make_v_dram = [&](const VDataType* data, index_t length) {
			const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
				data, // will update this pointer if using paged-kvcache
				make_tuple(length, kargs.hdim_v),
				make_tuple(kargs.stride_v, 1),
				number<FmlaPipeline::kAlignmentV>{},
				number<1>{});

			const auto v_dram_transposed =
				transform_tensor_view(v_dram_naive,
									  make_tuple(make_pass_through_transform(kargs.hdim_v),
												 make_pass_through_transform(length)),
									  make_tuple(sequence<1>{}, sequence<0>{}),
									  make_tuple(sequence<0>{}, sequence<1>{}));

			return pad_tensor_view(
				v_dram_transposed,
				make_tuple(number<FmlaPipeline::kN1>{}, number<FmlaPipeline::kK1>{}),
				sequence<kPadHeadDimV, kPadSeqLenK>{});
        };
        const auto v_dram = [&]() {
            if constexpr(kIsPagedKV)
            {
                return make_v_dram(nullptr, kargs.page_block_size);
            }
        };

        auto k_page_block_navigator_func = [&](int i_batch_) {
            if constexpr(kIsPagedKV)
            {
                const auto* block_indices =
                    reinterpret_cast<const int32_t*>(kargs.block_table_ptr) +
                    i_batch_ * kargs.batch_stride_block_table;
                const index_t num_blocks =
                    integer_divide_ceil(kv_l2p_offset + kargs.seqlen_k_ptr[i_batch_], kargs.page_block_size);

                const long_index_t fixed_offset =
                    static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_k;

                return make_page_block_navigator<const KDataType, 0>(
                    kargs.k_ptr,
                    kargs.batch_stride_k, // kcache page-block stride/size
                    fixed_offset,
                    block_indices,
                    num_blocks,
                    kargs.page_block_size,
                    k_dram,
                    make_k_dram(nullptr,
                                (kv_l2p_offset + kargs.seqlen_k_ptr[i_batch_]) -
                                    (num_blocks - 1) * kargs.page_block_size));
            }
        };

        auto v_page_block_navigator_func = [&](int i_batch_) {
            if constexpr(kIsPagedKV)
            {
                const auto* block_indices =
                    reinterpret_cast<const int32_t*>(kargs.block_table_ptr) +
                    i_batch_ * kargs.batch_stride_block_table;
                const index_t num_blocks =
                    integer_divide_ceil(kv_l2p_offset + seqlen_k_ptr[i_batch_], kargs.page_block_size);

                const long_index_t fixed_offset =
                    static_cast<long_index_t>(i_nhead_k) * kargs.nhead_stride_v;

                return make_page_block_navigator<const VDataType, 1>(
                    kargs.v_ptr,
                    kargs.batch_stride_v, // vcache page-block stride/size
                    fixed_offset,
                    block_indices,
                    num_blocks,
                    kargs.page_block_size,
                    v_dram,
                    make_v_dram(nullptr,
                                (kv_l2p_offset + seqlen_k_ptr[i_batch_]) -
                                    (num_blocks - 1) * kargs.page_block_size));
            }
        };

        auto q_dram_window_lengths = [&]() {
			if constexpr(FmlaPipeline::kQLoadOnce)
				return make_tuple(number<FmlaPipeline::kM0>{},
								  number<FmlaPipeline::kSubQKHeaddim>{});
			else
				return make_tuple(number<FmlaPipeline::kM0>{}, number<FmlaPipeline::kK0>{});
		}();

        auto k_dram_window_lengths =
            make_tuple(number<FmlaPipeline::kN0>{}, number<FmlaPipeline::kK0>{});
        auto v_dram_window_lengths =
            make_tuple(number<FmlaPipeline::kN1>{}, number<FmlaPipeline::kK1>{});

        // lse acc
        auto lse_acc_dram_window_func = [&, i_nhead_ = i_nhead](int i_split_) {
            constexpr auto lse_acc_dram_window_lengths = make_tuple(number<FmlaPipeline::kM0>{});
            LSEDataType* lse_acc_ptr = reinterpret_cast<LSEDataType*>(kargs.lse_acc_ptr) +
                                       static_cast<long_index_t>(i_nhead_) *
                                           (kMergeNumHeadGroupsSeqLenQ ? kargs.nhead_ratio_qk : 1) *
                                           kargs.nhead_stride_lse_acc +
                                       batch_offset_lse_acc + i_split_ * kargs.split_stride_lse_acc;

            const auto lse_acc_dram = [&] {
                const auto lse_acc_dram_naive = [&] {
                    if constexpr(kMergeNumHeadGroupsSeqLenQ)
                    {
                        // reshape: (nhead_ratio_qk, seqlen_q) -> (nhead_ratio_qk * seqlen_q)
                        const auto view = make_naive_tensor_view<address_space_enum::global>(
                            lse_acc_ptr,
                            make_tuple(kargs.nhead_ratio_qk, kargs.seqlen_q),
                            make_tuple(kargs.nhead_stride_lse_acc, 1),
                            number<1>{},
                            number<1>{});

                        return transform_tensor_view(view,
                                                     make_tuple(make_merge_transform(make_tuple(
                                                         kargs.nhead_ratio_qk, kargs.seqlen_q))),
                                                     make_tuple(sequence<0, 1>{}),
                                                     make_tuple(sequence<0>{}));
                    }
                    else
                    {
                        return make_naive_tensor_view<address_space_enum::global>(
                            lse_acc_ptr,
                            make_tuple(kargs.seqlen_q),
                            make_tuple(1),
                            number<1>{},
                            number<1>{});
                    }
                }();
                return pad_tensor_view(
                    lse_acc_dram_naive, lse_acc_dram_window_lengths, sequence<kPadSeqLenQ>{});
            }();

            return make_tile_window(lse_acc_dram, lse_acc_dram_window_lengths, {i_m0});
        };

        auto mask_func = [&](int seqlen_k_) {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmlaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    kargs.seqlen_q,
                    seqlen_k_,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmlaMask{kargs.seqlen_q, seqlen_k_};
        }();


		for(IndexDataType batch_id = begin_idx; batch_id <= end_idx; ++batch_id)
		{
			const IndexDataType n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
			kargs.seqlen_k = static_cast<index_t>(kargs.seqlen_k_ptr)[batch_id];
			const IndexDataType n_max_block = ck_tile::integer_divide_ceil(kargs.seqlen_k, FmlaPipeline::kM0);
			const IndexDataType n_block_min = batch_id == begin_idx ? begin_seqlen / FmlaPipeline::kM0 : 0;
			const IndexDataType n_block_max = batch_id == end_idx ? ck_tile::integer_divide_ceil(end_seqlen, FmlaPipeline::kM0) : n_max_block;

			const bool NoSplit = (n_block_min == 0 && n_block_max == n_max_block);
			ck_tile::index_t i_split = reinterpret_cast<IndexDataType*>(args.num_splits_ptr)[batch_id] + n_split_idx;

			auto lse_acc_dram_window = lse_acc_dram_window_func(i_split);
			auto q_dram_window = make_tile_window(
				q_dram,
				q_dram_window_lengths,
				{i_m0, 0});

			auto o_acc_tile = [&, i_split_ = i_split]() {
				if constexpr(kDoFp8StaticQuant)
				{
					return FmlaPipeline{}(q_dram_window,
										  identity{}, // q_element_func
										  k_dram_window_lengths,
										  k_page_block_navigator,
										  identity{}, // k_element_func
										  v_dram_window_lengths,
										  v_page_block_navigator,
										  identity{}, // v_element_func
										  lse_acc_dram_window,
										  identity{},            // lse_element_func
										  identity{},            // s_acc_element_func
										  scales{kargs.scale_p}, // p_compute_element_func
										  identity{},            // o_acc_element_func
                                          {
                                               n_max_block,
                                               n_block_min,
                                               n_block_max,
                                               i_split,
                                               i_block_m,
                                               i_nhead,
                                          },
										  i_split_,
										  mask(i_split),
										  kargs.scale_s,
										  smem_ptr);
				}
				else
				{
					return FmlaPipeline{}(q_dram_window,
										  k_dram_window_lengths,
										  k_page_block_navigator,
										  v_dram_window_lengths,
										  v_page_block_navigator,
										  bias_dram_window,
										  lse_acc_dram_window,
                                          {
                                               n_max_block,
                                               n_block_min,
                                               n_block_max,
                                               i_split,
                                               i_block_m,
                                               i_nhead,
                                          },
										  i_split_,
										  mask(i_split),
										  kargs.scale_s,
										  smem_ptr);
				}
			}();
			// Oacc DRAM and Oacc DRAM window
			auto o_acc_dram = [&] {
				const auto o_acc_dram_naive = [&] {
					return make_naive_tensor_view<address_space_enum::global>(
						o_acc_ptr,
						make_tuple(kargs.seqlen_q, kargs.hdim_v),
						make_tuple(kargs.stride_o_acc, 1),
						number<FmlaPipeline::kAlignmentOacc>{},
						number<1>{});
				}();

				return pad_tensor_view(
					o_acc_dram_naive,
					make_tuple(number<FmlaPipeline::kM0>{}, number<kHeadDimV>{}),
					sequence<kPadSeqLenQ, kPadHeadDimV>{});
			}();

			auto o_acc_dram_window =
				make_tile_window(o_acc_dram,
								 make_tuple(number<FmlaPipeline::kM0>{}, number<kHeadDimV>{}),
								 {i_m0, 0});
		}
	}
};
}
