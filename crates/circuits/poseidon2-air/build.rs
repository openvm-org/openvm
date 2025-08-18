use openvm_cuda_builder::{cuda_available, emit_cuda_cfg_if_available, CudaBuilder};

fn main() {
    emit_cuda_cfg_if_available();
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
        .file("cuda/src/dummy.cu")
        .build();

    builder.emit_link_directives(); // Call once at end
}
