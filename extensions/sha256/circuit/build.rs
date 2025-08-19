#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

#[cfg(feature = "cuda")]
fn main() {
    if !cuda_available() {
        return; // Skip CUDA compilation
    }

    let builder: CudaBuilder = CudaBuilder::new()
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
        .include("../../../crates/circuits/primitives/cuda/include")
        .include("../../../crates/circuits/sha256-air/cuda/include")
        .include("../../../crates/vm/cuda/src")
        .watch("cuda")
        .library_name("tracegen_gpu_primitives")
        .file("cuda/src/sha256.cu");

    builder.emit_link_directives();
    builder.build();
}

#[cfg(not(feature = "cuda"))]
fn main() {}
