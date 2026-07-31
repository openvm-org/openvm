#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }

        let builder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include("../circuits/primitives/cuda/include")
            .include("../circuits/poseidon2-air/cuda/include")
            .include("cuda/include")
            // CUB on recent CUDA toolchains can trigger a host-side GCC false positive here.
            .flag("-Xcompiler=-Wno-maybe-uninitialized");
        builder.emit_link_directives();

        builder
            .clone()
            .include("cuda/rvr/include")
            .watch("cuda/rvr/include")
            .library_name("tracegen_gpu_system")
            .files([
                "cuda/src/system/boundary.cu",
                "cuda/src/system/inventory.cu",
                "cuda/src/system/memory/merkle_tree.cu",
                "cuda/src/system/phantom.cu",
                "cuda/src/system/poseidon2.cu",
                "cuda/src/system/program.cu",
                "cuda/src/system/postflight.cu",
            ])
            .build();

        #[cfg(any(test, feature = "test-utils"))]
        {
            builder
                .clone()
                .library_name("tracegen_gpu_testing")
                .files_from_glob("cuda/src/testing/**/*.cu")
                .build();
        }
    }
}
