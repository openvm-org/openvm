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

        let mut system_builder = builder.clone().library_name("tracegen_gpu_system").files([
            "cuda/src/system/boundary.cu",
            "cuda/src/system/inventory.cu",
            "cuda/src/system/memory/merkle_tree.cu",
            "cuda/src/system/phantom.cu",
            "cuda/src/system/poseidon2.cu",
            "cuda/src/system/program.cu",
        ]);
        if cfg!(feature = "rvr") {
            system_builder = system_builder
                .include("cuda/rvr/include")
                .watch("cuda/rvr")
                .flag("-DOPENVM_RVR_REPLAY")
                .files([
                    "cuda/src/system/rvr_checkpoint_replay.cu",
                    "cuda/src/system/rvr_postflight.cu",
                ]);
        }
        system_builder.build();

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
