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
            .include("../../rv32-adapters/cuda/include")
            .watch("cuda/src/")
            .watch("../../../crates/circuits/primitives/cuda/include")
            .watch("../../../crates/vm/cuda/include")
            .watch("../../rv32-adapters/cuda/include")
            .library_name("tracegen_gpu_algebra")
            .file("cuda/src/is_eq.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
