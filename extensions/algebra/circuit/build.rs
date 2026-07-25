#[cfg(all(feature = "cuda", feature = "rvr"))]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    #[cfg(all(feature = "cuda", feature = "rvr"))]
    {
        if !cuda_available() {
            return;
        }

        let builder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include("../../../crates/circuits/primitives/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .include("../../../crates/vm/cuda/rvr/include")
            .include("cuda/include")
            .include("../../riscv/circuit/cuda/include")
            .include("../../riscv/circuit/cuda/rvr/include")
            .include("../../riscv-adapters/cuda/include")
            .watch("cuda")
            .library_name("tracegen_gpu_algebra")
            .file("cuda/src/field_expr_replay.cu")
            .file("cuda/src/modular_is_eq.cu")
            .file("cuda/src/modular_addsub_replay.cu")
            .file("cuda/src/vec_heap_projection.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
