use openvm_cuda_builder::{cuda_available, emit_cuda_cfg_if_available, CudaBuilder};

fn main() {
    emit_cuda_cfg_if_available();
    if !cuda_available() {
        return; // Skip CUDA compilation
    }

    let common = CudaBuilder::new()
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE") // Import headers
        .include("../primitives/cuda/include")
        .include("cuda/include")
        .include("../../../extensions/rv32-adapters/cuda/include")
        .include("../../../tracegen-gpu/cuda/src")
        .include("../poseidon2-air/cuda/include")
        .watch("cuda")
        .watch("src/cuda");

    common
        .clone()
        .library_name("tracegen_mod_builder")
        .files(["cuda/src/field_expression.cu"])
        .build();

    common.emit_link_directives(); // Call once at end
}
