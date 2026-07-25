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
            .include("../../../crates/circuits/poseidon2-air/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .include("cuda/include")
            .watch("cuda/src/count.cu")
            .watch("cuda/src/poseidon2.cu")
            .watch("cuda/src/call.cu")
            .watch("cuda/src/output.cu")
            .library_name("tracegen_gpu_deferral")
            .files(["cuda/src/count.cu", "cuda/src/poseidon2.cu"]);

        let builder = if std::env::var_os("CARGO_FEATURE_RVR").is_some() {
            builder
                .flag("-I../../../crates/vm/cuda/rvr/include")
                .watch("../../../crates/vm/cuda/rvr/include/arch/rvr/preflight.cuh")
                .watch("../../../crates/vm/cuda/rvr/include/arch/rvr/replay.cuh")
                .watch("cuda/rvr")
                .files(["cuda/rvr/call.cu", "cuda/rvr/output.cu"])
        } else {
            builder.files(["cuda/src/call.cu", "cuda/src/output.cu"])
        };

        builder.emit_link_directives();
        builder.build();
    }
}
