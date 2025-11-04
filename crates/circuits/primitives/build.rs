#[cfg(feature = "cuda")]
use std::{env, path::PathBuf};

#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }

        let builder: CudaBuilder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include("cuda/include")
            .watch("cuda")
            .library_name("tracegen_gpu_primitives")
            .files_from_glob("cuda/src/**/*.cu");

        builder.emit_link_directives();
        builder.build();

        // Export include dir for downstream crates:
        let include_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("cuda")
            .join("include");
        println!("cargo:include={}", include_path.display()); // -> DEP_CIRCUIT_PRIMITIVES_CUDA_INCLUDE
    }
}
