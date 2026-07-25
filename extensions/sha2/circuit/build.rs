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
            // Pass include roots without recursively watching them. The legacy SHA inputs stay
            // narrow, and checkpoint replay adds its own dependencies only under `rvr` below.
            .flag("-Icuda/include")
            .flag("-I../../../crates/circuits/primitives/cuda/include")
            .flag("-I../../../crates/vm/cuda/include")
            .watch("cuda/src/sha2_main.cu")
            .watch("cuda/src/sha2_hasher.cu")
            .watch("cuda/include/block_hasher")
            .watch("cuda/include/main")
            .watch("../../../crates/circuits/primitives/cuda/include/primitives/constants.h")
            .watch("../../../crates/circuits/primitives/cuda/include/primitives/encoder.cuh")
            .watch("../../../crates/circuits/primitives/cuda/include/primitives/execution.h")
            .watch("../../../crates/circuits/primitives/cuda/include/primitives/fp_array.cuh")
            .watch("../../../crates/circuits/primitives/cuda/include/primitives/histogram.cuh")
            .watch("../../../crates/circuits/primitives/cuda/include/primitives/less_than.cuh")
            .watch("../../../crates/circuits/primitives/cuda/include/primitives/trace_access.h")
            .watch("../../../crates/circuits/primitives/cuda/include/primitives/utils.cuh")
            .watch("../../../crates/vm/cuda/include/system/memory/controller.cuh")
            .watch("../../../crates/vm/cuda/include/system/memory/offline_checker.cuh")
            .watch("../../../crates/vm/cuda/include/system/memory/params.cuh")
            .library_name("tracegen_gpu_sha2");

        let builder = if std::env::var_os("CARGO_FEATURE_RVR").is_some() {
            builder
                .flag("-I../../../crates/vm/cuda/rvr/include")
                .watch("cuda/include/rvr/replay.cuh")
                .watch(
                    "../../../crates/circuits/primitives/cuda/include/primitives/buffer_view.cuh",
                )
                .watch("../../../crates/vm/cuda/rvr/include/arch/rvr/preflight.cuh")
                .watch("../../../crates/vm/cuda/rvr/include/arch/rvr/replay.cuh")
                .files(["cuda/src/rvr/sha2_main.cu", "cuda/src/rvr/sha2_hasher.cu"])
        } else {
            builder.files(["cuda/src/sha2_main.cu", "cuda/src/sha2_hasher.cu"])
        };

        builder.emit_link_directives();
        builder.build();
    }
}
