#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }

        // Build xorin CUDA library
        let xorin_builder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include("../../../crates/circuits/primitives/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .include("cuda/include")
            .watch("cuda")
            .watch("../../../crates/circuits/primitives/cuda")
            .watch("../../../crates/vm/cuda")
            .library_name("tracegen_gpu_xorin")
            .file("cuda/src/xorin.cu");

        xorin_builder.emit_link_directives();
        xorin_builder.build();
    }
}
