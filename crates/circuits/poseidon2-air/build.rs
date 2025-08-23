#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }

        let builder: CudaBuilder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE") // Import headers
            .include("cuda/include")
            .include("../primitives/cuda/include")
            .watch("cuda")
            .watch("../primitives/cuda")
            .library_name("tracegen_gpu_poseidon2_air")
            .file("cuda/src/dummy.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
