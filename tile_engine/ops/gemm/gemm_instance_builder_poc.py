# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
from pathlib import Path
import sys
import itertools
import json
from dataclasses import dataclass
from typing import List, Dict, Any

# Constants and mapping definitions
DATA_TYPE_MAP = {
    'fp32': 'float',
    'fp16': 'ck_tile::half_t',
    'bf16': 'ck_tile::bf16_t',
    'int8': 'ck_tile::int8_t',
    'fp8': 'ck_tile::fp8_t',
    'bf8': 'ck_tile::bf8_t',
    'int4': 'ck_tile::pk_int4_t'
}

LAYOUT_MAP = {
    'R': 'ck_tile::tensor_layout::gemm::RowMajor',
    'C': 'ck_tile::tensor_layout::gemm::ColumnMajor'
}

def BOOL_MAP(b: bool) -> str:
    return 'true' if b else 'false'

@dataclass
class GemmConfig:
    matrix_cfg: Dict[str, Any]
    impl_cfg: Dict[str, Any]
    
    @property
    def datatype(self) -> str:
        return self.matrix_cfg["datatype"]["values"][0]
    
    @property
    def layouts(self) -> List[str]:
        return [
            self.matrix_cfg["layout_a"]["values"][0],
            self.matrix_cfg["layout_b"]["values"][0],
            self.matrix_cfg["layout_c"]["values"][0]
        ]

class GemmCodeGenerator:
    def __init__(self, output_dir: str, config: GemmConfig):
        self.output_dir = Path(output_dir)
        self.config = config
        self.all_kernels = []
        
        # Validate configurations
        self._validate_config()

    def _validate_config(self):
        """Validate matrix and implementation configurations"""
        # Matrix config validation
        for param in ["datatype", "layout_a", "layout_b", "layout_c"]:
            if len(self.config.matrix_cfg[param]["values"]) != 1:
                raise ValueError(f"Matrix config {param} must have exactly one value")
        
        # Implementation traits validation
        required_params = ["tile_m", "tile_n", "tile_k", "warp_m", "warp_n", "warp_k",
                          "warp_tile_m", "warp_tile_n", "warp_tile_k", "pipeline",
                          "epilogue", "scheduler", "kPadM", "kPadN", "kPadK"]
        for param in required_params:
            if not self.config.impl_cfg.get(param, {}).get("values"):
                raise ValueError(f"Missing implementation parameter: {param}")

    def generate_all(self):
        """Generate all code components"""
        self.output_dir.mkdir(exist_ok=True)
        
        self._generate_common_header()
        self._generate_config_groups()
        self._generate_dispatcher()
        self._generate_main()

    def _generate_common_header(self):
        """Generate common header with datatypes and layouts"""
        content = f"""// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core.hpp"
#include "gemm_host.hpp"

// Data types
using ADataType = {DATA_TYPE_MAP[self.config.datatype]};
using BDataType = {DATA_TYPE_MAP[self.config.datatype]};
using AccDataType = float;
using CDataType = {DATA_TYPE_MAP[self.config.datatype]};

// Layout configurations
using ALayout = {LAYOUT_MAP[self.config.layouts[0]]};
using BLayout = {LAYOUT_MAP[self.config.layouts[1]]};
using CLayout = {LAYOUT_MAP[self.config.layouts[2]]};
"""
        (self.output_dir / "gemm_common.hpp").write_text(content)

    def _generate_config_groups(self):
        """Generate implementation configuration groups"""
        params = [
            ("pipeline", "pipeline"),
            ("epilogue", "epilogue"),
            ("scheduler", "scheduler"),
            ("kPadM", "kPadM"),
            ("kPadN", "kPadN"), 
            ("kPadK", "kPadK")
        ]
        
        # Generate all combinations
        for combo in itertools.product(*[self.config.impl_cfg[p]["values"] for (p, _) in params]):
            config = {name: value for (_, name), value in zip(params, combo)}
            self._generate_config_group(**config)

    def _generate_config_group(self, pipeline: str, epilogue: str, scheduler: str,
                              kPadM: bool, kPadN: bool, kPadK: bool):
        """Generate a configuration group with all tile/warp combinations"""
        group_name = f"{pipeline}_{epilogue}_{scheduler}_pad{BOOL_MAP(kPadM)}{BOOL_MAP(kPadN)}{BOOL_MAP(kPadK)}"
        filename = f"gemm_{group_name}.hpp"
        self.all_kernels.append(group_name)

        content = f"""// SPDX-License-Identifier: MIT
#include "gemm_common.hpp"

namespace {group_name} {{
"""
        # Add template struct with configuration
        content += self._generate_kernel_struct(pipeline, epilogue, scheduler, kPadM, kPadN, kPadK)
        
        # Add tile/warp instantiations
        tile_params = itertools.product(
            self.config.impl_cfg["tile_m"]["values"],
            self.config.impl_cfg["tile_n"]["values"],
            self.config.impl_cfg["tile_k"]["values"],
            self.config.impl_cfg["warp_m"]["values"],
            self.config.impl_cfg["warp_n"]["values"],
            self.config.impl_cfg["warp_k"]["values"],
            self.config.impl_cfg["warp_tile_m"]["values"],
            self.config.impl_cfg["warp_tile_n"]["values"],
            self.config.impl_cfg["warp_tile_k"]["values"]
        )

        for tile in tile_params:
            content += f"""
template struct GemmKernel<
    {tile[0]}, {tile[1]}, {tile[2]},  // Tile sizes
    {tile[3]}, {tile[4]}, {tile[5]},  // Warp counts
    {tile[6]}, {tile[7]}, {tile[8]}   // Warp tiles
>;"""

        content += f"\n}} // namespace {group_name}"
        (self.output_dir / filename).write_text(content)

    def _generate_kernel_struct(self, pipeline: str, epilogue: str, scheduler: str,
                               kPadM: bool, kPadN: bool, kPadK: bool) -> str:
        """Generate kernel struct template"""
        return f"""
template <int TileM, int TileN, int TileK,
          int WarpM, int WarpN, int WarpK,
          int WarpTileM, int WarpTileN, int WarpTileK>
struct GemmKernel {{
    static constexpr bool kPadM = {BOOL_MAP(kPadM)};
    static constexpr bool kPadN = {BOOL_MAP(kPadN)};
    static constexpr bool kPadK = {BOOL_MAP(kPadK)};
    
    static constexpr auto Pipeline = ck_tile::{pipeline}_pipeline;
    static constexpr auto Epilogue = ck_tile::{epilogue}_epilogue;
    static constexpr auto Scheduler = ck_tile::{scheduler}_scheduler;

    static float launch(ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s) {{
        // Kernel implementation using all template parameters
        // ... [Actual kernel launch code] ...
        return 0.0f; // Return actual timing
    }}
}};
"""

    def _generate_dispatcher(self):
        """Generate dispatch mechanism"""
        content = """// SPDX-License-Identifier: MIT
#include "gemm_common.hpp"
#include <unordered_map>

struct GemmDispatcher {
    static std::unordered_map<std::string, 
        float(*)(ck_tile::GemmHostArgs&, const ck_tile::stream_config&)> kernel_map;
    
    static void init() {
        if(!kernel_map.empty()) return;
        \n"""
        
        for group in self.all_kernels:
            content += f"""        kernel_map["{group}"] = [](ck_tile::GemmHostArgs& args, 
                                      const ck_tile::stream_config& s) {{
            return {group}::GemmKernel<>::launch(args, s);
        }};\n"""

        content += """    }
    
    static float dispatch(ck_tile::GemmHostArgs& args,
                         const ck_tile::stream_config& s) {
        init();
        const std::string key = assemble_key(args);
        if(auto it = kernel_map.find(key); it != kernel_map.end()) {
            return it->second(args, s);
        }
        throw std::runtime_error("No suitable kernel found: " + key);
    }

private:
    static std::string assemble_key(ck_tile::GemmHostArgs& args) {
        return std::string(args.pipeline) + "_" + 
               args.epilogue + "_" + 
               args.scheduler + "_" +
               (args.kPadM ? "T" : "F") +
               (args.kPadN ? "T" : "F") +
               (args.kPadK ? "T" : "F");
    }
}};

std::unordered_map<std::string, 
    float(*)(ck_tile::GemmHostArgs&, const ck_tile::stream_config&)> 
    GemmDispatcher::kernel_map;
"""
        (self.output_dir / "gemm_dispatcher.hpp").write_text(content)

    def _generate_main(self):
        """Generate main application file"""
        content = """// SPDX-License-Identifier: MIT
#include "gemm_common.hpp"
#include "gemm_dispatcher.hpp"

template <typename ADataType, typename BDataType,
          typename AccDataType, typename CDataType,
          typename ALayout, typename BLayout, typename CLayout>
float gemm_kernel_launch(ck_tile::GemmHostArgs& args, 
                        const ck_tile::stream_config& s) {
    return GemmDispatcher::dispatch(args, s);
}

// Include existing run implementation
#include "gemm_run_implementation.hpp"

int main(int argc, char* argv[]) {
    try {
        auto [result, parser] = create_args(argc, argv);
        if (!result) return EXIT_FAILURE;
        return run<ADataType, BDataType, AccDataType, CDataType,
                  ALayout, BLayout, CLayout>(parser);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\\n";
        return EXIT_FAILURE;
    }
}
"""
        (self.output_dir / "gemm_main.cpp").write_text(content)

def main():
    parser = argparse.ArgumentParser(description="GEMM Code Generator")
    parser.add_argument("-j", "--json", required=True, help="Config JSON file")
    parser.add_argument("-o", "--output", default="generated", help="Output directory")
    args = parser.parse_args()

    with open(args.json) as f:
        config_data = json.load(f)
    
    # Validate and parse configuration
    gemm_config = GemmConfig(
        matrix_cfg=config_data["matrix_configuration"],
        impl_cfg=config_data["implementation_traits"]
    )
    
    # Generate code
    generator = GemmCodeGenerator(args.output, gemm_config)
    generator.generate_all()

if __name__ == "__main__":
    main()