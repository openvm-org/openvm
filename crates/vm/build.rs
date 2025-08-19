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
            .include("../circuits/primitives/cuda/include")
            .include("../circuits/mod-builder/cuda/include")
            .include("../circuits/poseidon2-air/cuda/include")
            .watch("cuda/src/system")
            .library_name("tracegen_gpu_system")
            .files_from_glob("cuda/src/system/**/*.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
