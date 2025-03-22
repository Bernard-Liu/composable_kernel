# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
from enum import IntEnum
from pathlib import Path
import sys
from typing import List, Optional, Any
import functools
import itertools
import copy
import json
from dataclasses import dataclass

def get_if_str(idx, total, lase_else = True):
    if idx == 0:
        return 'if'
    elif idx < total - 1:
        return 'else if'
    else:
        if lase_else:
            return 'else'
        else:
            return 'else if'

 
DATA_TYPE_MAP = {'fp32'  : 'float',
                 'fp16'  : 'ck_tile::half_t',
                 'bf16'  : 'ck_tile::bf16_t',
                 'int8'  : 'ck_tile::int8_t',
                 'fp8'   : 'ck_tile::fp8_t',
                 'bf8'   :  'ck_tile::bf8_t',
                 'int4'  : 'ck_tile::pk_int4_t'
                }

LAYOUT_MAP = {'r' : 'ck_tile::tensor_layout::gemm::RowMajor',
              'c' : 'ck_tile::tensor_layout::gemm::ColumnMajor'}

PIPELINE_MAP = {'compv3' : 0,
                'compv4' : 1,
                'mem'    : 2}

SCHEDULER_MAP = {'interwave' : 0,
                 'intrawave' : 1}

EPILOGUE_MAP = {'default' : 0,
                'cshuffle' : 1}                 

def sizeOf(data_type):
    if data_type == 'fp16' or data_type == 'bf16':
        return 2
    elif data_type == 'int8' or data_type == 'fp8' or data_type == 'bf8':
        return 1
    elif data_type == 'int4': ## TODO:: needs to confirm
        return 0.5
    else:
        return 4

def BOOL_MAP(b_) -> str:
    if b_:
        return 'true'
    else:
        return 'false'

class gemm_instance_codegen:

    API_TRAITS_DEFINE = """
template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename CDataType_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          ck_tile::index_t tile_m_,
          ck_tile::index_t tile_n_,
          ck_tile::index_t tile_k_,
          ck_tile::index_t warp_m_,
          ck_tile::index_t warp_n_,
          ck_tile::index_t warp_k_,
          ck_tile::index_t warp_tile_m_,
          ck_tile::index_t warp_tile_n_,
          ck_tile::index_t warp_tile_k_,
          bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          int pipeline_,
          int scheduler_,
          int epilogue_>
struct gemm_traits_{
    using ADataType_ = ck_tile::remove_cvref_t<ADataType_>;
    using BDataType_ = ck_tile::remove_cvref_t<BDataType_>;
    using AccDataType_ = ck_tile::remove_cvref_t<AccDataType_>;
    using CDataType_ = ck_tile::remove_cvref_t<CDataType_>;
    using ALayout_ = ck_tile::remove_cvref_t<ALayout_>;
    using BLayout_ = ck_tile::remove_cvref_t<BLayout_>;
    using CLayout_ = ck_tile::remove_cvref_t<CLayout_>;
    
    static constexpr bool kPadM_ = kPadM_;
    static constexpr bool kPadN_ = kPadN_;
    static constexpr bool kPadK_ = kPadK_;
    
    using BlockTile_ = ck_tile::sequence<tile_m_, tile_n_, tile_k_>;
    using BlockWarps_ = ck_tile::sequence<warp_m_, warp_n_, warp_k_>;
    using WarpTile_ = ck_tile::sequence<warp_tile_m_, warp_tile_n_, warp_tile_k_>;
    
    bool PermuteA = false;
    bool PermuteB = false;

    using shape_ = ck_tile::TileGemmShape<BlockTile_, BlockWarps_, WarpTile_, PermuteA, PermuteB>;

    bool TransposeC = false;
    bool DoubleSmemBuffer = false;
    int kBlockPerCu                         = 1;
    ck_tile::index_t TileParitionerGroupNum = 8;
    ck_tile::index_t TileParitionerM01      = 4;

    int pipeline_ = pipeline_;
    int scheduler_ = scheduler_;
    int epilogue_ = epilogue_;
}

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename CDataType_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          ck_tile::index_t tile_m_,
          ck_tile::index_t tile_n_,
          ck_tile::index_t tile_k_,
          ck_tile::index_t warp_m_,
          ck_tile::index_t warp_n_,
          ck_tile::index_t warp_k_,
          ck_tile::index_t warp_tile_m_,
          ck_tile::index_t warp_tile_n_,
          ck_tile::index_t warp_tile_k_,
          bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          int pipeline_,
          int scheduler_,
          int epilogue_>
using traits_ = gemm_traits_<ADataType_, 
                             BDataType_, 
                             AccDataType_, 
                             CDataType_, 
                             ALayout_, 
                             BLayout_, 
                             CLayout_, 
                             tile_m_, 
                             tile_n_, 
                             tile_k_, 
                             warp_m_, 
                             warp_n_, 
                             warp_k_, 
                             warp_tile_m_, 
                             warp_tile_n_, 
                             warp_tile_k_, 
                             kPadM_, 
                             kPadN_, 
                             kPadK_, 
                             pipeline_, 
                             scheduler_, 
                             epilogue_>;

"""

    HOT_LOOP_FALSE = """
        if(tail_num == ck_tile::TailNumber::Full)
        {
            Run(ck_tile::bool_constant<false>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }
        else if(tail_num == ck_tile::TailNumber::Odd)
        {
            Run(ck_tile::bool_constant<false>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
        }
        else if(tail_num == ck_tile::TailNumber::Even)
        {
            Run(ck_tile::bool_constant<false>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
        }
        else
        {
            throw std::runtime_error("Num K loop must be larger than number of prefetech stages.");
        }  
"""
    RUN_MEM = """
            if(tail_num == ck_tile::TailNumber::One)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{});
            }
            else if(tail_num == ck_tile::TailNumber::Full)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }

            if constexpr(BaseGemmPipeline::PrefetchStages > 2)
            {
                if(tail_num == ck_tile::TailNumber::Two)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
                }
        
                if(tail_num == ck_tile::TailNumber::Three)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
                }
                if(tail_num == ck_tile::TailNumber::Four)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Four>{});
                }
                if(tail_num == ck_tile::TailNumber::Five)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Five>{});
                }
                if(tail_num == ck_tile::TailNumber::Six)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Six>{});
                }
                if(tail_num == ck_tile::TailNumber::Seven)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Seven>{});
                }
                throw std::runtime_error("The tile number is wrong! It should not exceed the prefetch stage numbers");
            }
"""

    RUN_COMPV3 = """
            if(tail_num == ck_tile::TailNumber::Full)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            else if(tail_num == ck_tile::TailNumber::Odd)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else if(tail_num == ck_tile::TailNumber::Even)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>{});
            }
            else
            {
                throw std::runtime_error("The tail number is wrong. It should be Full, Odd, or Even.");
            }
"""

    RUN_COMPV4 = """
            if(tail_num == ck_tile::TailNumber::Three)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
            }
            else
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
            }
"""

    COMMON_HEADER = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include <ck_tile/ops/epilogue.hpp>
#include <iostream>

#pragma once
{F_traits_define}
template<typename Traits_>
float gemm_kernel_launch_(ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
{{
    using ADataType = typename Traits_::ADataType_;
    using BDataType = typename Traits_::BDataType_;
    using AccDataType = typename Traits_::AccDataType_;
    using CDataType = typename Traits_::CDataType_;
    using ALayout = typename Traits_::ALayout_;
    using BLayout = typename Traits_::BLayout_;
    using CLayout = typename Traits_::CLayout_;
    using GemmShape = typename Traits_::shape_;
    
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   Traits_::TileParitionerGroupNum,
                                                   Traits_::TileParitionerM01>;

    using Traits  = 
        ck_tile::TileGemmTraits<Traits_::kPadM_,
                                Traits_::kPadN_,
                                Traits_::kPadK_,
                                ALayout,
                                BLayout,
                                CLayout>; 

    using GemmUniversalTraits = 
        ck_tile::TileGemmUniversalTraits<Traits_::kPadM_,
                                         Traits_::kPadN_,
                                         Traits_::kPadK_,
                                         Traits_::DoubleSmemBuffer,
                                         ALayout,
                                         BLayout,
                                         CLayout,
                                         Traits_::TransposeC>;    

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, 
                                     BDataType, 
                                     AccDataType, 
                                     GemmShape, 
                                     Traits>;      

    using compv3_BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>; 
    using compv4_BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<GemmPipelineProblem>;
    using mem_BaseGemmPipeline    = ck_tile::BaseGemmPipelineAgBgCrMem<GemmPipelineProblem>;

    using BaseGemmPipeline = 
        std::conditional_t<Traits_::pipeline_ == 0, compv3_BaseGemmPipeline,
        std::conditional_t<Traits_::pipeline_ == 1, compv4_BaseGemmPipeline, mem_BaseGemmPipeline>>;

    const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * GemmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    float ave_time{{0}};    
    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        using interwave_schedule = ck_tile::GemmPipelineScheduler::Interwave;
        using intrawave_schedule = ck_tile::GemmPipelineScheduler::Intrawave;
        using scheduler = std::conditional_t<Traits_::scheduler_, intrawave_schedule, interwave_schedule>;

        //constexpr auto scheduler      = ck_tile::GemmPipelineScheduler::Intrawave; ## TODO:: can I assign the value of conditional_t to constexpr auto ??
        
        using UniversalGemmProblem = 
            ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  GemmShape,
                                                  GemmUniversalTraits,
                                                  scheduler,
                                                  has_hot_loop_v,
                                                  tail_number_v>;  

        using compv3_GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>; 
        using compv4_GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<UniversalGemmProblem>;
        using mem_GemmPipeline    = ck_tile::GemmPipelineAgBgCrMem<UniversalGemmProblem>;
    
        using BaseGemmPipeline = 
            std::conditional_t<Traits_::pipeline_ == 0, compv3_GemmPipeline,
            std::conditional_t<Traits_::pipeline_ == 1, compv4_GemmPipeline, mem_GemmPipeline>>;
             
        using DefaultEpilogue = ck_tile::DefaultGemm2DEpilogue<
                                ck_tile::DefaultGemm2DEpilogueProblem<AccDataType, 
                                                                      CDatatType, 
                                                                      CLayout, 
                                                                      GemmConfig::kPadM,
                                                                      GemmConfig::kPadN,
                                                                      GemmConfig::M_Warp_Tile,
                                                                      GemmConfig::N_Warp_Tile,
                                                                      GemmConfig::K_Warp_Tile,
                                                                      UniversalGemmProblem::TransposeC>>;
        
        using CShuffleEpilogue = ck_tile::CShuffleEpilogue<
                            ck_tile::CShuffleEpilogueProblem<ADataType,
                                                             BDataType,
                                                             AccDataType,
                                                             CDataType,
                                                             CLayout,
                                                             GemmPipelineProblem::kBlockSize,
                                                             TilePartitioner::MPerBlock,
                                                             TilePartitioner::NPerBlock,
                                                             GemmConfig::M_Warp,
                                                             GemmConfig::N_Warp,
                                                             GemmConfig::M_Warp_Tile,
                                                             GemmConfig::N_Warp_Tile,
                                                             GemmConfig::K_Warp_Tile,
                                                             UniversalGemmProblem::TransposeC>>;

        using GemmEpilogue = std::conditional_t<Traits_::epilogue_, CShuffleEpilogue, DefaultEpilogue >;

        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {{
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!");
        }}

        if(s.log_level_ > 0)
        {{
            std::cout << "Launching kernel with args:"
                      << " grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                      << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                      << std::endl;
        }}

        ave_time = ck_tile::launch_kernel(s,
                                          ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                                              Kernel{{}}, grids, blocks, 0, kargs));
        return ave_time;
    }};     

    if(has_hot_loop){{
        if(Traits_::pipeline_ == "compv3"){{
            {run_compv3}
        }}else if (Traits_::pipeline_ == "compv4"){{
            {run_compv4}
        }}else{{
            {run_mem}
        }}
    }}else{{
        {hot_loop_false}
    }}
   
                                                                                                                    
    return ave_time;
}}
"""

    INSTANCE_BASE = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include <ck_tile/ops/epilogue.hpp>
#include <iostream>

{F_traits_define}


template<typename Traits_> 
float gemm_kernel_launch_(ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s);

template<typename ADataType, 
         typename BDataType, 
         typename AccDataType,
         typename CDataType,
         typename ALayout,         
         typename BLayout,
         typename CLayout>
float gemm_kernel_launch(ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
{{
    return gemm_kernel_launch_<traits_<{F_template}>>(args, s);
}}

"""

    @dataclass
    class datatype_configuration:
        F_ADataType: str
        F_BDataType: str
        F_AccDataType: str
        F_ODataType: str
    
    @dataclass
    class tile_shapes:
        F_BlockTile      : List[int]
        F_WarpPerBlock   : List[int]
        F_WarpTile       : List[int]

    @dataclass
    class gemm_kernel_method:
        F_pipeline        : Any
        F_epilogue        : Any
        F_scheduler       : Any

    @dataclass
    class gemm_traits:
        F_ADataType: str
        F_BDataType: str
        F_AccDataType: str
        F_ODataType: str
        F_A_Layout: str
        F_B_Layout: str
        F_C_Layout: str
        F_kPadM: bool
        F_kPadN: bool
        F_kPadK: bool
        tile_shapes: Any
        F_pipeline: str
        F_scheduler: str
        F_epilogue: str

        @property
        def trait_template(self) -> str:
            nnn = f'{DATA_TYPE_MAP[self.F_ADataType]}, {DATA_TYPE_MAP[self.F_BDataType]}, {DATA_TYPE_MAP[self.F_AccDataType]}, {DATA_TYPE_MAP[self.F_ODataType]}, '
            nnn+= f'{LAYOUT_MAP[self.F_A_Layout]}, {LAYOUT_MAP[self.F_B_Layout]}, {LAYOUT_MAP[self.F_C_Layout]}, '
            nnn += f'{self.tile_shapes.F_BlockTile[0]}, {self.tile_shapes.F_BlockTile[1]}, {self.tile_shapes.F_BlockTile[2]}, '
            nnn += f'{self.tile_shapes.F_WarpPerBlock[0]}, {self.tile_shapes.F_WarpPerBlock[1]}, {self.tile_shapes.F_WarpPerBlock[2]}, '
            nnn += f'{self.tile_shapes.F_WarpTile[0]}, {self.tile_shapes.F_WarpTile[1]}, {self.tile_shapes.F_WarpTile[2]}, '
            nnn += f'{self.F_kPadM}, {self.F_kPadN}, {self.F_kPadK}, '
            nnn += f'{PIPELINE_MAP[self.F_pipeline]}, {SCHEDULER_MAP[self.F_scheduler]}, {EPILOGUE_MAP[self.F_epilogue]}'
            return nnn
            
        @property
        def name(self) -> str:
            nnn = f'gemm_{self.F_ADataType}_{self.F_BDataType}_{self.F_ODataType}_{self.F_A_Layout}_{self.F_B_Layout}_{self.F_C_Layout}'
            nnn += f'_BT_{self.tile_shapes.F_BlockTile[0]}_{self.tile_shapes.F_BlockTile[1]}_{self.tile_shapes.F_BlockTile[2]}'
            nnn += f'_W_{self.tile_shapes.F_WarpPerBlock[0]}_{self.tile_shapes.F_WarpPerBlock[1]}_{self.tile_shapes.F_WarpPerBlock[2]}'
            nnn += f'_WT_{self.tile_shapes.F_WarpTile[0]}_{self.tile_shapes.F_WarpTile[1]}_{self.tile_shapes.F_WarpTile[2]}'
            nnn += f'_pad_{self.F_kPadM}_{self.F_kPadN}_{self.F_kPadK}'
            nnn += f'_{self.F_pipeline}_{self.F_scheduler}_{self.F_epilogue}'
            return nnn
    


    def __init__(self, working_path, json_data):
        self.working_path = working_path
        with open(json_data, 'r') as json_file:
            self.data = json.load(json_file)

    @property
    def name_common_header(self) -> str:
        return 'gemm_api_common'
    

    @property
    def common_header(self) -> str:
        str1 = self.COMMON_HEADER.format(
                F_traits_define = self.API_TRAITS_DEFINE,
                run_mem = self.RUN_MEM,
                run_compv3 = self.RUN_COMPV3,
                run_compv4 = self.RUN_COMPV4,
                hot_loop_false = self.HOT_LOOP_FALSE
                )
        return str1
    
    def content_api(self, template_name) -> str:
        str1 = self.INSTANCE_BASE.format(F_traits_define = self.API_TRAITS_DEFINE,
                                        F_template = template_name)
        return str1
        

    def get_blobs(self):
        gemm_traits = gemm_instance_codegen.gemm_traits
        tile_m = self.data['tile_m']["values"]
        tile_n = self.data['tile_n']["values"]
        tile_k = self.data['tile_k']["values"]

        warp_m = self.data['warp_m']["values"]  
        warp_n = self.data['warp_n']["values"]
        warp_k = self.data['warp_k']["values"]
        
        warp_tile_m = self.data['warp_tile_m']["values"]
        warp_tile_n = self.data['warp_tile_n']["values"]
        warp_tile_k = self.data['warp_tile_k']["values"]
        
        k_padm = self.data['kPadM']["values"]
        k_padn = self.data['kPadN']["values"]
        k_padk = self.data['kPadK']["values"]

        prec_type = self.data['datatype']["values"]

        a_layout = self.data['layout_a']["values"]
        b_layout = self.data['layout_b']["values"]
        c_layout = self.data['layout_c']["values"]

        pipeline = self.data['pipeline']["values"]
        scheduler = self.data['scheduler']["values"]
        epilogue = self.data['epilogue']["values"]

        blobs = []
        for i in range(len(tile_m)):
            ctype = prec_type[i]
            atype = prec_type[i]
            btype = prec_type[i]
            otype = 'fp32'
            if prec_type[i] in ['fp8', 'bf8']:
                ctype = 'fp16'
            elif prec_type[i] in ['int4']:
                atype = 'fp16'
                ctype = 'fp16'
            
            trait = gemm_traits(F_ADataType = atype,
                                F_BDataType = btype,
                                F_AccDataType = ctype,
                                F_ODataType = otype,
                                F_A_Layout = a_layout[i],
                                F_B_Layout = b_layout[i],
                                F_C_Layout = c_layout[i],
                                F_kPadM = k_padm[i],
                                F_kPadN = k_padn[i],
                                F_kPadK = k_padk[i],
                                tile_shapes = gemm_instance_codegen.tile_shapes(F_BlockTile = [tile_m[i], tile_n[i], tile_k[i]],
                                                                              F_WarpPerBlock = [warp_m[i], warp_n[i], warp_k[i]],
                                                                              F_WarpTile = [warp_tile_m[i], warp_tile_n[i], warp_tile_k[i]]),
                                F_pipeline = pipeline[i],
                                F_scheduler = scheduler[i],
                                F_epilogue = epilogue[i]
                                )
            blobs.append(trait)
        return blobs

    
    def list_blobs(self) -> None:
        w_p = Path(self.working_path)
        list_p = w_p / 'gemm_instance_blobs.txt'
        blobs = self.get_blobs()
        with list_p.open('w') as list_f:
            # api related file
            list_f.write(str(w_p / (self.name_common_header + ".hpp"))  + "\n")
            # kernel instance file
            for b in blobs:
               list_f.write(str(w_p / (b.name + ".cpp")) + "\n") 
        

    def gen_blobs(self) -> None:
        w_p = Path(self.working_path)
        (w_p / (self.name_common_header + ".hpp")).write_text(self.common_header)
        blobs = self.get_blobs()
        for b in blobs:
            w_str = self.content_api(b.trait_template)
            (w_p / (b.name + ".cpp")).write_text(w_str)

        
def do_list_blobs(args):
    #list out all the blobs
    generator = gemm_instance_codegen(args.working_path, args.json)
    generator.list_blobs()


def do_gen_blobs(args):
    generator = gemm_instance_codegen(args.working_path, args.json)
    generator.gen_blobs()


def main(args):
    if args.list_blobs:
        do_list_blobs(args)
    elif args.gen_blobs:
        do_gen_blobs(args)
    else:
        # If neither was specified, either do nothing or default to gen_blobs
        print("No mode specified (use --list_blobs or --gen_blobs). Generating by default...")
        do_gen_blobs(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm kernel",
    )
    parser.add_argument(
        "-w", "--working_path", default="./", required=False, help="the path where all the blobs are going to be generated"
    )
    parser.add_argument(
        "-j", "--json", required=True, help="Path to the json which contains the kernel configurations"
    )

    parser.add_argument(
        "-l",
        "--list_blobs",
        action='store_true',
        help="list all the kernels to a file, "
    )

    parser.add_argument(
        "-g",
        "--gen_blobs",
        action='store_true',
        help="generate all kernels into different files"
    )

    args = parser.parse_args()

    if (args.gen_blobs and args.list_blobs) or ((not args.gen_blobs) and (not args.list_blobs)):
        print('gen_blobs/list_blobs must specify only one option')
        sys.exit()

    p = Path(args.working_path)
    if not p.exists():
        p.mkdir()

    main(args)
