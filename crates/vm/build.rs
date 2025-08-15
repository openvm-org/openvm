use std::{env, process::Command};
use cc::Build;

fn capture_glob(b: Build, pattern: &str) -> Build {
    let mut b = b;
    for path in glob::glob(pattern).expect("Invalid glob pattern").flatten() {
        if path.is_file() {
            b.file(path);
        }
    }
    b
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
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rerun-if-changed=src/cuda");
        println!("cargo:rerun-if-env-changed=CUDA_ARCH");
        println!("cargo:rerun-if-env-changed=CUDA_OPT_LEVEL");
        println!("cargo:rerun-if-env-changed=CUDA_DEBUG");

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
        let mut common = Build::new();
        common
            .cuda(true)
            // Include paths
            .include("cuda")
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
        common.include("../circuits/primitives/cuda/include");
        common.include("../cuda-includes-to-delete");

        capture_glob(common.clone(), "cuda/**/*.cu").compile("tracegen_gpu_system");

        // Make sure CUDA and our utilities are linked
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cuda");
    }
}
