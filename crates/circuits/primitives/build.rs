use openvm_cuda_builder::{cuda_available, emit_cuda_cfg_if_available, CudaBuilder};

fn main() {
    emit_cuda_cfg_if_available();
    if !cuda_available() {
        return; // Skip CUDA compilation
    }

    let common: CudaBuilder = CudaBuilder::new()
        .include_from_dep("DEP_CUDA_COMMON_INCLUDE") // Import headers
        .include("cuda/include")
        .watch("cuda")
        .watch("src/cuda_abi.rs");

    common
        .clone()
        .library_name("tracegen_gpu_primitives")
        .files(["cuda/src/bitwise_op_lookup.cu"])
        .files(["cuda/src/range_tuple.cu"])
        .files(["cuda/src/var_range.cu"])
        .files(["cuda/src/dummy/bitwise_op_lookup.cu"])
        .files(["cuda/src/dummy/encoder.cu"])
        .files(["cuda/src/dummy/fibair.cu"])
        .files(["cuda/src/dummy/is_equal.cu"])
        .files(["cuda/src/dummy/is_zero.cu"])
        .files(["cuda/src/dummy/less_than.cu"])
        .files(["cuda/src/dummy/range_tuple.cu"])
        .files(["cuda/src/dummy/var_range.cu"])
        .build();

    common.emit_link_directives(); // Call once at end
}
