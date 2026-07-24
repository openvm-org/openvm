#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }

        let mut builder: CudaBuilder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include("cuda/include")
            .include("../../../crates/circuits/primitives/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .include("../../riscv-adapters/cuda/include")
            .watch("../../../crates/circuits/primitives/cuda")
            .watch("../../riscv-adapters/cuda")
            .watch("cuda/src")
            .library_name("tracegen_gpu_rv64im")
            .files_from_glob("cuda/src/**/*.cu");

        if cfg!(feature = "rvr") {
            builder = builder
                .include("cuda/rvr/include")
                .include("../../../crates/vm/cuda/rvr/include")
                .watch("cuda/rvr/src")
                .flag("-DOPENVM_RVR_REPLAY");
        }

        builder.emit_link_directives();
        builder.build();
    }
}
