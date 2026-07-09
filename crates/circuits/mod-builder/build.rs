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
            .include("../primitives/cuda/include")
            .include("../../vm/cuda/include")
            .watch("cuda")
            .watch("../primitives/cuda")
            .library_name("tracegen_gpu_mod_builder")
            .file("cuda/src/field_expr.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
