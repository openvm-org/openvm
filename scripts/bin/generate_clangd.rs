use std::{env, fs};

use openvm_scripts::{find_cuda_include_dirs, get_cuda_dep_common_include_dirs};
use serde::Serialize;

#[derive(Serialize)]
struct CompileFlags {
    #[serde(rename = "Add")]
    add: Vec<String>,
}

#[derive(Serialize)]
struct Diagnostics {
    #[serde(rename = "UnusedIncludes")]
    unused_includes: String,
    #[serde(rename = "MissingIncludes")]
    missing_includes: String,
}

#[derive(Serialize)]
struct ClangdConfig {
    #[serde(rename = "CompileFlags")]
    compile_flags: CompileFlags,
    #[serde(rename = "Diagnostics")]
    diagnostics: Diagnostics,
}

fn main() -> eyre::Result<()> {
    let workspace_root = env::current_dir()?;
    println!(
        "Generating .clangd for workspace: {}",
        workspace_root.display()
    );

    let include_dirs = find_cuda_include_dirs(&workspace_root);
    let common_include_dirs = get_cuda_dep_common_include_dirs();

    println!("Found {} include directories:", include_dirs.len());
    for dir in &include_dirs {
        match dir.strip_prefix(&workspace_root) {
            Ok(rel) => println!("  - {}", rel.display()),
            Err(_) => println!("  - {}", dir.display()),
        }
    }

    let compile_flags: Vec<String> = include_dirs
        .iter()
        .chain(common_include_dirs.iter())
        .map(|p| format!("-I{}", p.display()))
        .chain([
            "-x".into(),
            "cuda".into(),
            "-std=c++17".into(),
            "--cuda-gpu-arch=sm_70".into(),
            "-D__CUDA_ARCH__=700".into(),
        ])
        .collect();

    let config = ClangdConfig {
        compile_flags: CompileFlags { add: compile_flags },
        diagnostics: Diagnostics {
            unused_includes: "Strict".into(),
            missing_includes: "Strict".into(),
        },
    };

    let yaml = serde_yaml::to_string(&config)?;
    let output_path = workspace_root.join(".clangd");
    fs::write(&output_path, yaml)?;

    println!(
        "\n✅ .clangd file generated successfully at {}",
        output_path.display()
    );
    println!("   Total include directories: {}", include_dirs.len());
    println!("   Configuration will apply to all .cu and .cuh files in the repository");

    Ok(())
}
