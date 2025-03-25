// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename SaccDataType_,
          typename PDataType_,
          typename LSEDataType_,
          typename OaccDataType_,
          typename ODataType_,
          typename BlockFmlaShape_,
          bool kIsGroupMode_,
          typename FmlaMask_,
          typename Traits_>
struct BlockFmlaFwdSplitKVPipelineProblem
{
    using QDataType             = remove_cvref_t<QDataType_>;
    using KDataType             = remove_cvref_t<KDataType_>;
    using VDataType             = remove_cvref_t<VDataType_>;
    using SaccDataType          = remove_cvref_t<SaccDataType_>;
    using LSEDataType           = remove_cvref_t<LSEDataType_>;
    using PDataType             = remove_cvref_t<PDataType_>;
    using OaccDataType          = remove_cvref_t<OaccDataType_>;
    using ODataType             = remove_cvref_t<ODataType_>;
    using BlockFmlaShape        = remove_cvref_t<BlockFmlaShape_>;
    using FmlaMask              = remove_cvref_t<FmlaMask_>;
    using Traits                = remove_cvref_t<Traits_>;

    static constexpr index_t kNumGemm0Warps = BlockFmlaShape::NumGemm0Warps;
    static constexpr index_t kNumGemm1Warps = BlockFmlaShape::NumGemm1Warps;
    static constexpr index_t kBlockSize     = BlockFmlaShape::NumWarps * get_warp_size();

    static constexpr bool kIsGroupMode = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ       = Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = Traits::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = Traits::kPadHeadDimV;
    static constexpr bool kStoreLSE         = Traits::kStoreLSE;
    static constexpr bool kHasDropout       = Traits::kHasDropout;
    static constexpr bool kDoFp8StaticQuant = Traits::kDoFp8StaticQuant;
    static constexpr index_t kBlockPerCu    = Traits::kBlockPerCu;
};
