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
            .include("../../cuda-includes")
            .include("../circuits/primitives/cuda/include")
            .watch("cuda/system")
            .library_name("tracegen_gpu_system")
            .files_from_glob("cuda/system/**/*.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
