#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }

        println!("cargo:rerun-if-env-changed=OPENVM_RVR_CUDA_G2_ONLY");
        let mut builder: CudaBuilder = CudaBuilder::new()
            .include_from_dep("DEP_CUDA_COMMON_INCLUDE")
            .include("cuda/include")
            .include("../../../crates/circuits/primitives/cuda/include")
            .include("../../../crates/vm/cuda/include")
            .include("../../riscv-adapters/cuda/include")
            .watch("cuda")
            .watch("../../../crates/circuits/primitives/cuda")
            .watch("../../../crates/vm/cuda")
            .watch("../../riscv-adapters/cuda")
            .library_name("tracegen_gpu_rv64im")
            .files_from_glob("cuda/src/**/*.cu");

        if std::env::var("OPENVM_RVR_CUDA_G2_ONLY").as_deref() == Ok("1") {
            builder = builder
                .flag("-DOPENVM_RVR_CUDA_G2_ONLY=1")
                .flag("-Xcompiler=-Wno-unused-parameter");
        }

        builder.emit_link_directives();
        builder.build();
    }
}
