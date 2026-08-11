#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    #[cfg(feature = "cuda")]
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
            .include("../../riscv-adapters/cuda/include")
            .watch("cuda")
            .watch("../../../crates/circuits/primitives/cuda/include")
            .watch("../../../crates/vm/cuda/include")
            .watch("../../../crates/vm/cuda/rvr/include/arch/rvr/preflight.cuh")
            .watch("../../../crates/vm/cuda/rvr/include/arch/rvr/replay.cuh")
            .watch("../../riscv/circuit/cuda/include")
            .watch("../../riscv-adapters/cuda/include")
            .library_name("tracegen_gpu_algebra")
            .file("cuda/src/field_expr_replay.cu")
            .file("cuda/src/modular_is_eq.cu")
            .file("cuda/src/modular_addsub_replay.cu")
            .file("cuda/src/vec_heap_projection.cu")
            .file("cuda/src/ec_mul_projection.cu")
            .file("cuda/src/ec_mul_tracegen.cu");

        builder.emit_link_directives();
        builder.build();
    }
}
