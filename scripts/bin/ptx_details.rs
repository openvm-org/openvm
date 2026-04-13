use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use clap::Parser;
use openvm_scripts::{find_cuda_include_dirs, get_cuda_dep_common_include_dirs};
use tempfile::Builder as TempfileBuilder;

#[derive(Parser)]
#[command(author, version, about = "Compile CUDA code with nvcc")]
struct Args {
    /// Write output to this path. If omitted, a temporary file is used and deleted.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Path to nvcc (default: nvcc from PATH)
    #[arg(long, default_value = "nvcc")]
    nvcc: String,

    /// Path to CUDA installation (used for include paths)
    #[arg(long, env = "CUDA_PATH", default_value = "/usr/local/cuda")]
    cuda_path: String,

    /// CUDA architecture (e.g. 89 -> sm_89 / compute_89)
    #[arg(long, default_value = "89")]
    cuda_arch: String,

    /// nvcc thread count (-t)
    #[arg(long, default_value_t = 16)]
    threads: u32,

    /// Input .cu file to compile
    input: PathBuf,
}

fn add_if_exists(cmd: &mut Command, include_dir: &Path) {
    if include_dir.exists() {
        cmd.arg(format!("-I{}", include_dir.display()));
    }
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();
    let workspace_root = env::current_dir()?;

    // Reuse the same include discovery approach as other scripts.
    let mut include_dirs = find_cuda_include_dirs(&workspace_root);
    include_dirs.extend(get_cuda_dep_common_include_dirs());

    let (out_path, _tmp_file_guard) = match args.out {
        Some(p) => (p, None),
        None => {
            let tmp = TempfileBuilder::new().suffix(".ptx").tempfile()?;
            let path = tmp.path().to_path_buf();
            (path, Some(tmp))
        }
    };

    let cuda_include = PathBuf::from(&args.cuda_path).join("include");
    let cccl_include = cuda_include.join("cccl");

    let mut cmd = Command::new(&args.nvcc);
    cmd.env("LC_ALL", "C");

    // Keep flags matched to the original ptx.sh as closely as possible.
    cmd.args([
        "-ccbin=c++",
        "-Xcompiler",
        "-O3",
        "-Xcompiler",
        "-ffunction-sections",
        "-Xcompiler",
        "-fdata-sections",
        "-Xcompiler",
        "-fPIC",
        "-m64",
    ]);

    for dir in &include_dirs {
        cmd.arg(format!("-I{}", dir.display()));
    }

    add_if_exists(&mut cmd, &cuda_include);
    add_if_exists(&mut cmd, &cccl_include);

    cmd.args([
        "-Xcompiler",
        "-Wall",
        "-Xcompiler",
        "-Wextra",
        "--std=c++17",
        "--expt-relaxed-constexpr",
        "-Xfatbin=-compress-all",
    ]);

    cmd.arg("-gencode")
        .arg(format!("arch=compute_{a},code=sm_{a}", a = args.cuda_arch));
    cmd.arg("-gencode").arg(format!(
        "arch=compute_{a},code=compute_{a}",
        a = args.cuda_arch
    ));
    cmd.arg(format!("-t{}", args.threads));

    cmd.args(["--ptxas-options=-v", "-O3"]);
    cmd.arg("-o").arg(&out_path);
    cmd.args(["-c", "--device-c"]);

    // Make input absolute (so callers can run from workspace root like other scripts).
    let input = if args.input.is_absolute() {
        args.input
    } else {
        workspace_root.join(args.input)
    };
    cmd.arg(input);

    let status = cmd.status()?;
    if !status.success() {
        return Err(eyre::eyre!("nvcc failed with status: {status}"));
    }

    Ok(())
}
