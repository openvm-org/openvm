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
            .include("../../riscv/circuit/cuda/include")
            .include("../../riscv-adapters/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .watch("cuda")
            .watch("../../../crates/circuits/primitives/cuda")
            .watch("../../riscv/circuit/cuda")
            .watch("../../riscv-adapters/cuda")
            .watch("../../../crates/vm/cuda")
            .library_name("tracegen_gpu_bigint")
            .file("cuda/src/bigint.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
