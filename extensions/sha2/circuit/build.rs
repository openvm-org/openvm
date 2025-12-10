#[cfg(feature = "cuda")]
use openvm_cuda_builder::{cuda_available, CudaBuilder};

fn main() {
    #[cfg(feature = "cuda")]
    {
        if !cuda_available() {
            return; // Skip CUDA compilation
        }
        // Hybrid (CPU->GPU) path for SHA-2 uses CPU tracegen and does not require CUDA kernels
        // while the new design is being ported. We still keep the build script for compatibility,
        // but skip compiling any .cu files.
        let _builder: CudaBuilder = CudaBuilder::new();
    }
}
