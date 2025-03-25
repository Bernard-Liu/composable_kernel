// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmla/pipeline/block_fmla_pipeline_qr_ks_vs_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// mla qkv are all located in LDS
template <typename Problem_, typename Policy_ = BlockFmlaPipelineQRKSVSDefaultPolicy>
struct BlockFmlaPipelineQRKSVS
{
    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KVDataType            = remove_cvref_t<typename Problem::KVDataType>;
    using SaccDataType          = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType   = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType             = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType          = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using BlockFmlaShape             = remove_cvref_t<typename Problem::BlockFmlaShape>;
    using FmlaMask              = remove_cvref_t<typename Problem::FmlaMask>;

	static constexpr bool kQLoadOnce = true;
	static_assert(kQLoadOnce == Policy::QLoadOnce);

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = BlockFmlaShape::kM0;
    static constexpr index_t kN0           = BlockFmlaShape::kN0;
    static constexpr index_t kK0           = BlockFmlaShape::kK0;
    static constexpr index_t kN1           = BlockFmlaShape::kN1;
    static constexpr index_t kK1           = BlockFmlaShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockFmlaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockFmlaShape::kSubQKHeaddim; //TODO: need to think that do we need to split kv? for the fmha think kSubQKHeaddim should be less than 256
    static constexpr index_t kVHeaddim     = BlockFmlaShape::kVHeaddim;
// if K headdim split the v should be split for share weight;
// split kv should have some pozzles.

	static constexpr bool kPadSeqLenQ = Problem::kPadSeqLenQ;
	static constexpr bool kPadSeqLenK = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = Problem::kPadHeadDimV;
    static constexpr bool kStoreLSE    = Problem::kStoreLSE;


	static constexpr index_t kAlignmentQ =
		kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
	static constexpr index_t kAlignmentK =
		kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
	static constexpr index_t kAlignmentV =
		kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>(); // V is share weight with K, for -1 dim [:512]

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();

	static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
		else
			return 1;
	}();

    static constexpr const char* name = "qr mla";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename LSEElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowLengths& k_dram_block_window_lengths, // N0*K0 tile
               const KPageBlockNavigator& k_page_block_navigator,
               const KElementFunction& k_element_func,
               const VDramBlockWindowLengths& v_dram_block_window_lengths, // N1*K1 tile
               const VPageBlockNavigator& v_page_block_navigator,
               const VElementFunction& v_element_func,
               LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp, // M0*1 tile
               const LSEaccElementFunction& lse_acc_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               index_t i_split,
               FmlaMask mask,
               float scale_s,
               void* smem_ptr) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");
        // Q tile in LDS
		QDataType* q_lds_ptr =
			static_cast<QDataType*>(static_cast<void*>(static_cast<char*>(smem_ptr)));
		auto q_lds = make_tensor_view<address_space_enum::lds>(
			q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());

		KVDataType* kv_lds_ptr = static_cast<KVDataType*>(static_cast<void*>(
			static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQ<Problem>()));
		auto k_lds = make_tensor_view<address_space_enum::lds>(
			kv_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
		auto k_lds_window =
			make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});
		auto v_lds = make_tensor_view<address_space_enum::lds>(
			kv_lds_ptr, Policy::template MakeVLdsBlockDescriptor<Problem>());
		auto v_lds_window =
			make_tile_window(k_lds, make_tuple(number<kN1>{}, number<kK1>{}), {0, 0});

		PDataType* p_lds_ptr = static_cast<PDataType*>(static_cast<void*>(
			static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeOffsetP<Problem>()));
		auto p_lds = make_tensor_view<address_space_enum::lds>(
			p_lds_ptr, Policy::template MakePLdsBlockDescriptor<Problem>());

        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

		int wave_group_idx = GetWaveGroupIdx();
		//TODO: add calculate in wave_group_1
		if (wave_group_idx == 0)
		{
			constexpr auto gemm_2 = Policy::template GetKVBlockGemm<Problem>();
			q
		}
		else
		{
			
		}
	}

};

}
