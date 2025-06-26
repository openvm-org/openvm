use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=../backend/cuda/include");
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_OPT_LEVEL");

    // Get CUDA_ARCH from environment or detect it
    let cuda_arch = match env::var("CUDA_ARCH") {
        Ok(arch) => arch,
        Err(_) => {
            // Run nvidia-smi command to get arch
            let output = Command::new("nvidia-smi")
                .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
                .output()
                .expect("Failed to execute nvidia-smi");

            let arch = String::from_utf8(output.stdout)
                .unwrap()
                .trim()
                .replace(".", ""); // Convert "7.5" to "75"

            // Set environment variable for future builds
            println!("cargo:rustc-env=CUDA_ARCH={}", arch);
            arch
        }
    };
    println!("cargo:rerun-if-changed=build.rs");

    let opt_level = env::var("CUDA_OPT_LEVEL").unwrap_or("3".to_string());

    let mut builder = cc::Build::new();
    builder
        .cuda(true)
        // Include paths
        .include("../backend/cuda/include")
        .include("cuda/include")
        .include("cuda/src")
        // CUDA specific flags
        .flag("--std=c++17")
        .flag("--expt-relaxed-constexpr")
        .flag("--device-link")
        // Compute capability for T4
        .flag("-gencode")
        .flag(format!("arch=compute_{},code=sm_{}", cuda_arch, cuda_arch));

    // Add optimization flags based on build type
    if env::var("DEBUG").unwrap_or_default() == "true" {
        builder.flag("-O0").flag("-g");
    } else {
        builder.flag(format!("--ptxas-options=-O{}", opt_level));
    }
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        builder.include(format!("{}/include", cuda_path));
    }

    builder
        .file("cuda/src/extensions/rv32im/auipc.cu")
        .file("cuda/src/extensions/rv32im/hintstore.cu")
        .file("cuda/src/extensions/rv32im/jalr.cu")
        .file("cuda/src/primitives/bitwise_op_lookup.cu")
        .file("cuda/src/primitives/var_range.cu")
        .file("cuda/src/primitives/range_tuple.cu")
        .file("cuda/src/dummy/dummy_chip.cu")
        .file("cuda/src/dummy/range_tuple.cu")
        .file("cuda/src/dummy/encoder.cu")
        .file("cuda/src/dummy/fibair.cu")
        .file("cuda/src/dummy/less_than.cu")
        .file("cuda/src/dummy/is_zero.cu")
        .file("cuda/src/dummy/is_equal.cu")
        .file("cuda/src/dummy/poseidon2.cu")
        .file("cuda/src/dummy/utils.cu")
        .file("cuda/src/system/poseidon2.cu")
        .file("cuda/src/testing/execution.cu")
        .file("cuda/src/testing/memory.cu")
        .file("cuda/src/testing/program.cu")
        .compile("tracegen_gpu");

    // Make sure CUDA and our utilities are linked
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
}
