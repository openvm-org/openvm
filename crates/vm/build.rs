use openvm_cuda_builder::{cuda_available, emit_cuda_cfg_if_available, CudaBuilder};

fn main() {
    emit_cuda_cfg_if_available();
    if !cuda_available() {
        return; // Skip CUDA compilation
    }

    let common = CudaBuilder::new()
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE")  // Import headers
        .include("../../cuda-includes")
        .include("../circuits/primitives/cuda/include")
        // TODO: we need to reference the new ones? Like crates/circuits/primitives/cuda/include?
        .watch("cuda")
        .watch("src/cuda");

    common
        .clone()
        .library_name("tracegen_gpu_system")
        .files([
            "cuda/public_values.cu",
            "cuda/program.cu",
            "cuda/memory/merkle_tree.cu",
            "cuda/boundary.cu",
            "cuda/poseidon2.cu",
            "cuda/access_adapters.cu",
            "cuda/phantom.cu",
        ])
        .build();

    common.emit_link_directives(); // Call once at end
}
