use openvm_cuda_builder::{cuda_available, emit_cuda_cfg_if_available, CudaBuilder};

fn main() {
    emit_cuda_cfg_if_available();
    if !cuda_available() {
        return; // Skip CUDA compilation
    }

    let common = CudaBuilder::new()
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE")  // Import headers
        .watch("cuda")
        .watch("src/cuda");

    common.clone()
        .library_name("tracegen_gpu_system")
        .files([
            "cuda/execution.cuh",
            "cuda/public_values.cu",
            "cuda/program.cu",
            "cuda/program.cuh",
            "cuda/native_adapter.cuh",
            "cuda/memory",
            "cuda/memory/controller.cuh",
            "cuda/memory/merkle_tree.cu",
            "cuda/memory/offline_checker.cuh",
            "cuda/memory/address.cuh",
            "cuda/boundary.cu",
            "cuda/poseidon2.cu",
            "cuda/access_adapters.cu",
            "cuda/phantom.cu"
        ])
        .build();
    
    common.emit_link_directives(); // Call once at end
}
