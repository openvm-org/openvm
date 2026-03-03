#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }

        let builder: CudaBuilder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include("cuda/include")
            .include("../../../crates/circuits/primitives/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .watch("cuda")
            .watch("../../../crates/circuits/primitives/cuda")
            .watch("../../../crates/vm/cuda")
            .library_name("tracegen_gpu_rv32im")
            .files_from_glob("cuda/src/**/*.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
