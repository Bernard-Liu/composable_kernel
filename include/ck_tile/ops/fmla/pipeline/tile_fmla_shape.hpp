// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockTile_, // sequence<...
          typename Gemm0BlockWarps_,
          typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_,
          bool IsVLayoutRowMajor_>
struct TileFmlaShape
{
	using BlockTile = remove_cvref_t<BlockTile_>;
	using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
	using Gemm0WarpTile = remove_cvref_t<Gemm0WarpTile_>;
	using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
	using Gemm1WarpTile = remove_cvref_t<Gemm1WarpTile_>;
	
	static constexpr auto I1 = number<1>{};

	static constexpr index_t NumGemm0Warps = 
		reduce_on_sequenc(Gemm0BlockWarps{}, multiplies{}, I1);
	static constexpr index_t NumGemm1Warps =
		reduce_on_sequenc(Gemm1BlockWarps{}, multiplies{}, I1);
    static_assert(NumGemm1Warps % NumGemm0Warps == 0);

	static constexpr index_t NumWarps = max(NumGemm0Warps, NumGemm1Warps);

    static constexpr index_t kM0 = BlockTile::at(number<0>{}); // tile size along q seqlen
    static constexpr index_t kN0 = BlockTile::at(number<1>{}); // tile size along k seqlen
    static constexpr index_t kK0 = BlockTile::at(number<2>{}); // tile size along qk gemm unroll
    static constexpr index_t kN1 = BlockTile::at(number<3>{}); // tile size along v head_dim
    static constexpr index_t kK1 = BlockTile::at(number<4>{}); // tile size along kv gemm unroll
    static constexpr index_t kQKHeaddim =
        BlockTile::at(number<5>{}); // total length of K0, used for pipeline that need load Q at
                                    // once (or repeately load Q as a whole tile)
    static constexpr index_t kVHeaddim =
        BlockTile::at(number<5>{}); // total length of N1, for vheaddim not equal qkheaddim
};
}
