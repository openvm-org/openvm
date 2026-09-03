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
            // each translation unit watches only the headers it consumes.
            .flag("-I../../riscv/circuit/cuda/include")
            .include("../../riscv-adapters/cuda/include")
            .flag("-I../../../crates/vm/cuda/include")
            .watch("../../../crates/circuits/primitives/cuda")
            .watch("../../riscv/circuit/cuda/include/riscv/cores")
            .watch("../../riscv-adapters/cuda")
            .watch("../../../crates/vm/cuda/include/system/memory")
            .library_name("tracegen_gpu_bigint")
            .flag("-I../../../crates/vm/cuda/rvr/include")
            .watch("../../../crates/vm/cuda/rvr/include/arch/rvr/preflight.cuh")
            .watch("../../../crates/vm/cuda/rvr/include/arch/rvr/replay.cuh")
            .file("cuda/src/bigint_replay.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
