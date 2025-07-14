use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=../backend/cuda/include");
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_OPT_LEVEL");
    println!("cargo:rerun-if-changed=build.rs");

    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
            .output()
            .expect("Failed to run nvidia-smi");

        let arch = String::from_utf8(output.stdout)
            .unwrap()
            .trim()
            .replace(".", "");
        println!("cargo:rustc-env=CUDA_ARCH={}", arch);
        arch
    });

    let opt_level = env::var("CUDA_OPT_LEVEL").unwrap_or_else(|_| "3".to_string());
    let is_debug = env::var("DEBUG").unwrap_or_default() == "true";

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
        // Compute capability for T4
        .flag("-gencode")
        .flag(format!("arch=compute_{},code=sm_{}", cuda_arch, cuda_arch));

    if is_debug {
        common.flag("-O0").flag("-g");
    } else {
        common.flag(format!("--ptxas-options=-O{}", opt_level));
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
            .file("cuda/src/extensions/native/fri/fri.cu")
            .file("cuda/src/extensions/native/poseidon2/kernels.cu")
            .compile("tracegen_gpu_native");
    }
    {
        let mut b = common.clone();
        b.file("cuda/src/extensions/algebra/modular_chip/is_eq.cu")
            .compile("tracegen_gpu_algebra");
    }
    {
        let mut b = common.clone();
        b.file("cuda/src/extensions/keccak256/keccak256.cu")
            .file("cuda/src/extensions/keccak256/keccakf.cu")
            .compile("tracegen_gpu_keccak");
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
        b.file("cuda/src/system/boundary.cu")
            .file("cuda/src/system/phantom.cu")
            .file("cuda/src/system/poseidon2.cu")
            .file("cuda/src/system/program.cu")
            .file("cuda/src/system/public_values.cu")
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
