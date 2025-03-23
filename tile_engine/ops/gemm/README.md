We should have a generated file system look like as:

generated/
├── gemm_fp16_RCR_common.hpp              # Datatype + layouts
├── gemm_fp16_RCR_memory_cshuffle.hpp     # All tile/warp configs for memory pipeline + cshuffle
├── gemm_fp16_RCR_compv3_default.hpp      # All tile/warp configs for compv3 + default epilogue
├── gemm_fp16_RCR_compv4_cshuffle.hpp     # All tile/warp configs for compv4 + cshuffle
└── gemm_fp16_RCR_dispatcher.hpp          # Dispatch logic

If we have multiple datatypes we could have:
generated/
├── gemm_fp16_RCR/
│   ├── tensor_configuration.hpp
│   ├── gemm_fp16_RCR_var0.hpp
│   ├── gemm_fp16_RCR_var1.hpp
│   └── gemm_fp16_RCR_main.cpp
├── gemm_int4_CRC/
│   ├── tensor_configuration.hpp
│   ├── gemm_int4_CRC_var0.hpp
│   └── ...

After the compilation, we should put the pipeline and epilogue, .... to the runtime to select out the generated instance files.
Or we could have the benchmark file like following.
 def generate_benchmark_main(self):
        content = """// SPDX-License-Identifier: MIT
        #include "gemm_host.hpp"
        #include <chrono>

        template<int KernelID>
        void benchmark_kernel(ck_tile::GemmHostArgs& args) {
            auto start = std::chrono::high_resolution_clock::now();
            GemmKernel<KernelID>::run(args);
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "Kernel " << KernelID << " time: " 
                    << std::chrono::duration<double>(end - start).count() << "s\\n";
        }
        """
We could also provide a log:
    for kernel_id in range(NUM_KERNELS):
        run(f"./gemm_benchmark {kernel_id} > results/{kernel_id}.log")

In each .hpp files we already included all of the tile sizes for the certain pipeline+epilogues+scheduler. We also should select the best for the different pipeline+epiloges+scheduler combinations. We could also benchmark in the runtime level. When we run the certain executable. Each gemm_fp16_RCR and gemm_int4_CRC should have an executable.