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
            // These include roots are passed without recursively watching them:
            // the legacy TU uses only the core and memory subtrees below, while
            // the RVR-only headers are watched only when that feature is enabled.
            .flag("-I../../riscv/circuit/cuda/include")
            .include("../../riscv-adapters/cuda/include")
            .flag("-I../../../crates/vm/cuda/include")
            .watch("../../../crates/circuits/primitives/cuda")
            .watch("../../riscv/circuit/cuda/include/riscv/cores")
            .watch("../../riscv-adapters/cuda")
            .watch("../../../crates/vm/cuda/include/system/memory")
            .library_name("tracegen_gpu_bigint")
            .file("cuda/src/bigint.cu");

        let builder = if std::env::var_os("CARGO_FEATURE_RVR").is_some() {
            builder
                .flag("-I../../riscv/circuit/cuda/rvr/include")
                .flag("-I../../../crates/vm/cuda/rvr/include")
                .watch("../../riscv/circuit/cuda/rvr/include/riscv/replay.cuh")
                .watch("../../../crates/vm/cuda/rvr/include/arch/rvr/preflight.cuh")
                .file("cuda/src/bigint_replay.cu")
        } else {
            builder
        };

        builder.emit_link_directives();
        builder.build();
    }
}
