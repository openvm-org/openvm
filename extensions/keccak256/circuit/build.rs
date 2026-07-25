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
            .include("../../../crates/circuits/primitives/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .include("cuda/include")
            .watch("cuda/src")
            .library_name("tracegen_gpu_keccak256")
            .files_from_glob("cuda/src/*.cu");

        let builder = if std::env::var_os("CARGO_FEATURE_RVR").is_some() {
            builder
                .flag("-I../../../crates/vm/cuda/rvr/include")
                .watch("../../../crates/vm/cuda/rvr/include/arch/rvr/preflight.cuh")
                .watch("../../../crates/vm/cuda/rvr/include/arch/rvr/replay.cuh")
                .watch("cuda/rvr")
                .files_from_glob("cuda/rvr/*.cu")
        } else {
            builder
        };

        builder.emit_link_directives();
        builder.build();
    }
}
