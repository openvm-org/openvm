#[cfg(feature = "cuda")]
use {
    openvm_cuda_builder::{CudaBuilder, cuda_available},
    std::process::exit,
};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            eprintln!("cargo:warning=CUDA is not available");
            exit(1);
        }

        let common = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include_from_dep("DEP_CIRCUIT_PRIMITIVES_CUDA_INCLUDE")
            .include("cuda/include")
            .include("../cuda-backend/cuda/include")
            .include("../../openvm/crates/circuits/poseidon2-air/cuda/include");

        common.emit_link_directives();

        common
            .clone()
            .library_name("cuda-recursion-v2")
            .files_from_glob("cuda/src/**/*.cu")
            .build();
    }
}
