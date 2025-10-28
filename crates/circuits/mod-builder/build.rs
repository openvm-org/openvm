#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }

        let builder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE") // Import headers
            .include("../primitives/cuda/include")
            .include("cuda/include")
            .include("../../../extensions/rv32-adapters/cuda/include")
            .include("../poseidon2-air/cuda/include")
            .include("../../vm/cuda/include")
            .watch("src/cuda_abi.rs")
            .watch("cuda")
            .watch("../primitives/cuda/include")
            .watch("../../../extensions/rv32-adapters/cuda/include")
            .watch("../poseidon2-air/cuda/include")
            .watch("../../vm/cuda")
            .library_name("tracegen_mod_builder")
            .file("cuda/src/field_expression.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
