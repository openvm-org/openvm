use std::{
    path::{Path, PathBuf},
    process::Command,
};

use clap::{arg, Parser};
use openvm_scripts::{
    find_cuda_include_dirs, find_files_with_extension, get_cuda_dep_common_include_dirs,
};

#[derive(Parser)]
#[command(author, version, about = "Run Clang-Tidy on CUDA files")]
pub struct Args {
    /// Root directory to search
    directory: PathBuf,

    /// Path to clang-tidy (default: clang-tidy from PATH)
    #[arg(long, default_value = "clang-tidy")]
    clang_tidy: String,

    /// Path to CUDA installation
    #[arg(long, default_value = "/usr/local/cuda")]
    cuda_path: String,

    /// CUDA architecture to use
    #[arg(long, default_value = "80")]
    cuda_arch: String,
}

fn run_clang_tidy_single(
    file: &Path,
    clang_tidy: &str,
    cuda_path: &str,
    cuda_arch: &str,
    includes: &[String],
) -> eyre::Result<bool> {
    let mut cmd = Command::new(clang_tidy);

    cmd.args([
        "-warnings-as-errors='*'",
        "-header-filter='.*'",
        "-extra-arg=-Wno-unknown-cuda-version",
    ])
    .arg(file.to_str().unwrap())
    .args(["--", "-x", "cuda", "-std=c++17"])
    .arg(format!("--cuda-path={}", cuda_path))
    .arg(format!("--cuda-gpu-arch=sm_{}", cuda_arch))
    .arg("-D__CUDACC__")
    .arg(format!("-I{}/include", cuda_path));

    // Check if CCCL directory exists (CUDA 13.0+)
    let cccl_path = PathBuf::from(cuda_path).join("include").join("cccl");
    if cccl_path.exists() {
        cmd.arg(format!("-I{}", cccl_path.display()));
    }

    for include in includes {
        cmd.arg(format!("-I{}", include));
    }

    println!("Running for {}", file.display());

    let status = cmd.status()?;
    Ok(status.success())
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();
    let include_dirs = find_cuda_include_dirs(&args.directory)
        .iter()
        .chain(get_cuda_dep_common_include_dirs().iter())
        .map(|p| p.to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let files = find_files_with_extension(&args.directory, "cu");

    if files.is_empty() {
        println!("No .cu files found.");
        return Ok(());
    }

    let mut failures = Vec::new();

    for file in &files {
        match run_clang_tidy_single(
            file,
            &args.clang_tidy,
            &args.cuda_path,
            &args.cuda_arch,
            &include_dirs,
        ) {
            Ok(true) => {}
            Ok(false) => failures.push(file),
            Err(e) => {
                eprintln!("Error running clang-tidy on {}: {}", file.display(), e);
                failures.push(file);
            }
        }
    }

    println!("\n=== Summary ===");
    println!(
        "Total: {}  Succeeded: {}  Failed: {}",
        files.len(),
        files.len() - failures.len(),
        failures.len()
    );

    if !failures.is_empty() {
        println!("Failed files:");
        for file in failures {
            println!("  {}", file.display());
        }
        return Err(eyre::eyre!("Failed files found"));
    }
    Ok(())
}
