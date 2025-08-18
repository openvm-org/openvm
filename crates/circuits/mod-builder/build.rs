#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

#[cfg(feature = "cuda")]
fn main() {
    if !cuda_available() {
        return; // Skip CUDA compilation
    }

    let builder = CudaBuilder::new()
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE") // Import headers
        .include("../primitives/cuda/include")
        .include("cuda/include")
        .include("../../../extensions/rv32-adapters/cuda/include")
        .include("../poseidon2-air/cuda/include")
        .include("../../../tracegen-gpu/cuda/src") // TODO[arayi]: change once system is migrated
        .watch("src/cuda_abi.rs")
        .watch("cuda")
        .watch("../primitives/cuda/include")
        .watch("../../../extensions/rv32-adapters/cuda/include")
        .watch("../poseidon2-air/cuda/include")
        .watch("../../../tracegen-gpu/cuda/src") // TODO[arayi]: change once system is migrated
        .library_name("tracegen_mod_builder")
        .file("cuda/src/field_expression.cu");

    builder.emit_link_directives();
    builder.build();
}

#[cfg(not(feature = "cuda"))]
fn main() {}
