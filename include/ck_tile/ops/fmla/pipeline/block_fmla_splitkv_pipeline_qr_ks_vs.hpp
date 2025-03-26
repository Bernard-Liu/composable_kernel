// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmla/pipeline/block_fmla_pipeline_qr_ks_vs_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#include <cwchar>

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
    static constexpr index_t kNThreadsS    = BlockFmlaShape::kNThreadsS;
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

    struct FmlaPipelineMetaData
    {
        ck_tile::index_t n_max_block;
        ck_tile::index_t n_block_min;
        ck_tile::index_t n_block_max;
        ck_tile::index_t i_split;
        ck_tile::index_t i_block_m;
        ck_tile::index_t i_nhead;
        bool NoSplit;
    };

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
               FmlaPipelineMetaData metadata,
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

        int n_block = n_block_max - 1;

        KVDataType* kv_lds_ptr = static_cast<KVDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeKV<Problem>()));
        auto k_lds = make_tensor_view<address_space_enum::lds>(
            kv_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_ld_lds_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});
        auto k_st_lds_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            kv_lds_ptr, Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_ld_lds_window =
            make_tile_window(k_lds, make_tuple(number<kN1>{}, number<kK1>{}), {0, 0});
        auto v_st_lds_window =
            make_tile_window(k_lds, make_tuple(number<kN1>{}, number<kK1>{}), {0, 0});

        PDataType* p_lds_ptr = static_cast<PDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeOffsetP<Problem>()));
        auto p_lds = make_tensor_view<address_space_enum::lds>(
            p_lds_ptr, Policy::template MakePLdsBlockDescriptor<Problem>());
        auto p_lds_window =
            make_tile_window(p_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});
        auto p_ld_lds_window =
            make_tile_window(p_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});
        auto p_st_lds_window =
            make_tile_window(p_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});

        PDataType* scale_lds_ptr = static_cast<PDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeOffsetScale<Problem>()));
        auto scale_lds = make_tensor_view<address_space_enum::lds>(
            scale_lds_ptr, Policy::template MakeScaleLdsBlockDescriptor<Problem>());
        auto scale_ld_lds_window =
            make_tile_window(scale_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});
        auto scale_st_lds_window =
            make_tile_window(scale_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});

        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1 / 2;

        constexpr auto gemm_1 = Policy::template GetKVBlockGemm1<Problem>();

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        int wave_group_idx = GetWaveGroupIdx();
        auto o_acc = OaccBlockTileType{};
        clear_tile(o_acc);

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};
        //TODO: add calculate in wave_group_1
        if (wave_group_idx == 0)
        {
            constexpr auto gemm_0 = Policy::template GetKVBlockGemm2<Problem>();
            auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                                  q_dram_block_window_tmp.get_window_lengths(),
                                                  q_dram_block_window_tmp.get_window_origin(),
                                                  Policy::template MakeQRegTileDistribution<Problem>());

            auto q = load_tile(q_dram_window);
            auto q_tile = tile_elementwise_in(q_element_func, q);

            using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
            auto s_acc              = SaccBlockTileType{};
            const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
            const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };
            using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(s_acc));

            if(n_block % 2 == 1)
            {
                move_tile_window(k_lds_window, {kN0 / 8, 0});
                move_tile_window(v_lds_window, {kN0 / 8, 0});
            }

            for(; n_block >= n_block_min; --n_block)
            {
                block_sync_lds();
                gemm_0(s_acc,
                       get_slice_tile(q_tile,
                                      sequence<0, i_k0 * kK0>{},
                                      sequence<kM0, (i_k0 + 1) * kK0>{}),
                       k_lds_window);
                block_sync_lds();

                const auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}
                auto m_local = block_tile_reduce<SMPLComputeDataType>(
                    s,
                    sequence<1>{},
                    f_max,
                    -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
                block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

                const auto m_old = m; // m{j-1}
                tile_elementwise_inout(
                    [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

                static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                    /// NOTICE: bias might be materialized mask including -inf values, need
                    /// consideration
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 FmhaMask::IsMasking)
                    {
                        return raw_m == -numeric<SMPLComputeDataType>::infinity()
                                   ? type_convert<SMPLComputeDataType>(0.f)
                                   : raw_m;
                    }
                    else
                    {
                        return raw_m;
                    }
                };

                auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                    s.get_tile_distribution()); // Pcompute{j}
                sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    auto row_max = scale_s * get_validated_m(m[i_idx]);
#endif
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                     BiasEnum == BlockAttentionBiasEnum::ALIBI)
                        {
                            p_compute(i_j_idx) = exp2(s[i_j_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);
                        }
#else
                        p_compute(i_j_idx)     = exp(s[i_j_idx] - get_validated_m(m[i_idx]));
#endif
                    });
                });

                auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                    p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

                block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
                // l{j}, Oacc{j}
                constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
                sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    const auto tmp = [&]() {
                        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                     BiasEnum == BlockAttentionBiasEnum::ALIBI)
                        {
                            return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            auto row_max = scale_s * get_validated_m(m[i_idx]);
                            return exp2(scale_s * m_old[i_idx] - row_max);
                        }
                    }();
#else
                    const auto tmp       = exp(m_old[i_idx] - get_validated_m(m[i_idx]));
#endif
                    l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
                    sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        // FIXME: this use different equation from FA v2 paper,
                        // but produce correc result.
                        // Is the equation wrong?
                        o_acc(i_j_idx) *= tmp;
                    });
                });

                block_sync_lds();
                gemm_1(o_acc,
                       get_slice_tile(
                           p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                       v_lds_window);
                block_sync_lds();
            }
		}
		else
		{
            //TODO: aligned_physical_seqlen_k_start is not confirmed
            auto [i_page_block_k, k_dram_block_window] = k_page_block_navigator.make_tile_window(
                k_dram_block_window_lengths, {aligned_physical_seqlen_k_start, 0});
			
            auto [i_page_block_v, v_dram_window] = v_page_block_navigator.make_tile_window(
                v_dram_block_window_lengths,
                {0, aligned_physical_seqlen_k_start},
                Policy::template MakeVDramTileDistribution<Problem>());

            auto k_dram_window = make_tile_window(
                k_dram_block_window,
                Policy::template MakeKDramTileDistribution<Problem>()); // K DRAM tile window for
            if(n_block % 2 == 1)
            {
                move_tile_window(k_lds_window, {kN0, 0});
                move_tile_window(v_lds_window, {kN0 / 8, 0});
            }
            auto k_block_tile = load_tile(k_dram_window);
            move_tile_window(k_dram_window, {0, kK0});
            store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
#pragma unroll 1
            for(;n_block >= metadata.n_block_min; --n_block)
            {
                if(n_block - 1 >= n_block_min)
                {
                    auto k_block_tile = load_tile(k_dram_window);
                    move_tile_window(k_dram_window, {0, kK0});
                    store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                    block_lds_sync();
                }
                p = load_tile(p_lds_window);
                // tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, acc);
                block_sync_lds();
                gemm_1(o_acc,
                       get_slice_tile(
                           p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                       v_lds_window);
                block_sync_lds();
            }
		}
	}

};

}
