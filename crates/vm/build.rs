use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    if !cuda_available() {
        return; // Skip CUDA compilation
    }

    let common = CudaBuilder::new()
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE") // Import headers
        .include("../../cuda-includes")
        .include("../circuits/primitives/cuda/include")
        .watch("cuda/system");

    common
        .clone()
        .library_name("tracegen_gpu_system")
        .files_from_glob("cuda/system/**/*.cu")
        .build();

    common.emit_link_directives(); // Call once at end
}
