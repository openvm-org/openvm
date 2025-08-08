use std::{env, process::Command};

// Glob helper to watch multiple files
fn watch_glob(pattern: &str) {
    for path in glob::glob(pattern).expect("Invalid glob pattern").flatten() {
        if path.is_file() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

// Detect optimal NVCC parallel jobs
fn nvcc_parallel_jobs() -> String {
    // Try to detect CPU count from std
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    // Allow override from NVCC_THREADS env var
    let threads = std::env::var("NVCC_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(threads);

    format!("-t{}", threads)
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../backend/cuda/include");
    println!("cargo:rerun-if-changed=../backend/src/cuda");
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_OPT_LEVEL");
    println!("cargo:rerun-if-env-changed=CUDA_DEBUG");
    watch_glob("src/**/cuda*");

    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .expect("Failed to run nvidia-smi");

        let full_output = String::from_utf8(output.stdout).unwrap();
        let arch = full_output
            .lines()
            .next()
            .expect("`nvidia-smi --query-gpu=compute_cap` failed to return any output")
            .trim()
            .replace('.', "");
        println!("cargo:rustc-env=CUDA_ARCH={}", arch);
        arch
    });

    // CUDA_DEBUG shortcut
    if env::var("CUDA_DEBUG").map(|v| v == "1").unwrap_or(false) {
        env::set_var("CUDA_OPT_LEVEL", "0");
        env::set_var("CUDA_LAUNCH_BLOCKING", "1");
        env::set_var("CUDA_MEMCHECK", "1");
        env::set_var("RUST_BACKTRACE", "1");
        println!("cargo:warning=CUDA_DEBUG=1 → forcing CUDA_OPT_LEVEL=0, CUDA_LAUNCH_BLOCKING=1, CUDA_MEMCHECK=1,RUST_BACKTRACE=1");
    }

    // Get CUDA_OPT_LEVEL from environment or use default value
    // 0 → No optimization (fast compile, debug-friendly)
    // 1 → Minimal optimization
    // 2 → Balanced optimization (often same as -O3 for some kernels)
    // 3 → Maximum optimization (usually default for release builds)
    let cuda_opt_level = env::var("CUDA_OPT_LEVEL").unwrap_or_else(|_| "3".to_string());

    // Common CUDA settings
    let mut common = cc::Build::new();
    common
        .cuda(true)
        // Include paths
        .include("../backend/cuda/include")
        .include("cuda/include")
        .include("cuda/src")
        // CUDA specific flags
        .flag("--std=c++17")
        .flag("-Xfatbin=-compress-all")
        .flag("--expt-relaxed-constexpr")
        // .flag("--device-link")
        // Compute capability
        .flag("-gencode")
        .flag(format!("arch=compute_{},code=sm_{}", cuda_arch, cuda_arch))
        .flag(nvcc_parallel_jobs());

    if cuda_opt_level == "0" {
        common.debug(true);
        common.flag("-O0");
    } else {
        common.debug(false);
        common.flag(format!("--ptxas-options=-O{}", cuda_opt_level));
    }

    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        common.include(format!("{}/include", cuda_path));
    }

    {
        let mut b = common.clone();
        b.file("cuda/src/extensions/rv32im/auipc.cu")
            .file("cuda/src/extensions/rv32im/alu.cu")
            .file("cuda/src/extensions/rv32im/shift.cu")
            .file("cuda/src/extensions/rv32im/less_than.cu")
            .file("cuda/src/extensions/rv32im/mul.cu")
            .file("cuda/src/extensions/rv32im/hintstore.cu")
            .file("cuda/src/extensions/rv32im/load_sign_extend.cu")
            .file("cuda/src/extensions/rv32im/loadstore.cu")
            .file("cuda/src/extensions/rv32im/jalr.cu")
            .file("cuda/src/extensions/rv32im/divrem.cu")
            .file("cuda/src/extensions/rv32im/blt.cu")
            .file("cuda/src/extensions/rv32im/beq.cu")
            .file("cuda/src/extensions/rv32im/jal_lui.cu")
            .file("cuda/src/extensions/rv32im/mulh.cu")
            .compile("tracegen_gpu_rv32im");
    }

    {
        let mut b = common.clone();
        b.file("cuda/src/extensions/native/field_arithmetic.cu")
            .file("cuda/src/extensions/native/branch_eq.cu")
            .file("cuda/src/extensions/native/castf.cu")
            .file("cuda/src/extensions/native/loadstore.cu")
            .file("cuda/src/extensions/native/fri/fri.cu")
            .file("cuda/src/extensions/native/field_extension/field_extension.cu")
            .file("cuda/src/extensions/native/poseidon2/kernels.cu")
            .file("cuda/src/extensions/native/jal_rangecheck.cu")
            .compile("tracegen_gpu_native");
    }

    {
        let mut b = common.clone();
        b.file("cuda/src/extensions/algebra/modular_chip/is_eq.cu")
            .compile("tracegen_gpu_algebra");
    }

    {
        let mut b = common.clone();
        b.file("cuda/src/mod_builder/field_expression.cu")
            .compile("tracegen_mod_builder");
    }

    {
        let mut b = common.clone();
        b.file("cuda/src/extensions/keccak256/keccak256.cu")
            .file("cuda/src/extensions/keccak256/keccakf.cu")
            .compile("tracegen_gpu_keccak");
    }
    {
        let mut b = common.clone();
        b.file("cuda/src/extensions/sha256/sha256.cu")
            .compile("tracegen_gpu_sha256");
    }

    {
        let mut b = common.clone();
        b.file("cuda/src/extensions/bigint.cu")
            .compile("tracegen_gpu_bigint");
    }
    {
        let mut b = common.clone();
        b.file("cuda/src/primitives/bitwise_op_lookup.cu")
            .file("cuda/src/primitives/var_range.cu")
            .file("cuda/src/primitives/range_tuple.cu")
            .compile("tracegen_gpu_primitives");
    }
    {
        let mut b = common.clone();
        b.file("cuda/src/dummy/bitwise_op_lookup.cu")
            .file("cuda/src/dummy/encoder.cu")
            .file("cuda/src/dummy/fibair.cu")
            .file("cuda/src/dummy/less_than.cu")
            .file("cuda/src/dummy/is_zero.cu")
            .file("cuda/src/dummy/is_equal.cu")
            .file("cuda/src/dummy/poseidon2.cu")
            .file("cuda/src/dummy/range_tuple.cu")
            .file("cuda/src/dummy/var_range.cu")
            .compile("tracegen_gpu_dummy");
    }
    {
        let mut b = common.clone();
        b.file("cuda/src/system/access_adapters.cu")
            .file("cuda/src/system/boundary.cu")
            .file("cuda/src/system/phantom.cu")
            .file("cuda/src/system/poseidon2.cu")
            .file("cuda/src/system/program.cu")
            .file("cuda/src/system/public_values.cu")
            .file("cuda/src/system/memory/merkle_tree.cu")
            .compile("tracegen_gpu_system");
    }
    {
        let mut b = common.clone();
        b.file("cuda/src/testing/execution.cu")
            .file("cuda/src/testing/memory.cu")
            .file("cuda/src/testing/program.cu")
            .compile("tracegen_gpu_testing");
    }

    // Make sure CUDA and our utilities are linked
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
}
