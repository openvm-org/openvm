use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
};

use serde::Serialize;
use walkdir::WalkDir;

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

fn is_cuda_path(path: &Path) -> bool {
    path.to_string_lossy().to_lowercase().contains("cuda")
}

fn has_cuda_files(path: &Path) -> bool {
    WalkDir::new(path)
        .into_iter()
        .filter_map(Result::ok)
        .any(|e| {
            e.file_type().is_file()
                && matches!(
                    e.path().extension().and_then(|s| s.to_str()),
                    Some("cuh") | Some("cu")
                )
        })
}

fn find_cuda_include_dirs(workspace_root: &Path) -> Vec<PathBuf> {
    let mut include_dirs: BTreeSet<PathBuf> = BTreeSet::new();

    for entry in WalkDir::new(workspace_root)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_dir() && e.file_name() == "include")
    {
        let include_dir = entry.path().to_path_buf();

        if include_dir.components().any(|c| c.as_os_str() == "target") {
            continue;
        }

        if is_cuda_path(&include_dir) || has_cuda_files(&include_dir) {
            include_dirs.insert(include_dir);
        }
    }

    include_dirs.into_iter().collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workspace_root = env::current_dir()?;
    println!(
        "Generating .clangd for workspace: {}",
        workspace_root.display()
    );

    let include_dirs = find_cuda_include_dirs(&workspace_root);

    println!("Found {} include directories:", include_dirs.len());
    for dir in &include_dirs {
        match dir.strip_prefix(&workspace_root) {
            Ok(rel) => println!("  - {}", rel.display()),
            Err(_) => println!("  - {}", dir.display()),
        }
    }

    let mut all_includes: Vec<String> = include_dirs
        .iter()
        .map(|p| format!("-I{}", p.display()))
        .collect();

    // Add DEP_CUDA_COMMON_INCLUDE provided at compile-time via build.rs
    if let Some(val) = option_env!("DEP_CUDA_COMMON_INCLUDE") {
        for p in env::split_paths(val) {
            if !p.as_os_str().is_empty() {
                all_includes.push(format!("-I{}", p.display()));
            }
        }
    }

    let mut compile_flags = all_includes;
    compile_flags.extend([
        "-x".into(),
        "cuda".into(),
        "-std=c++17".into(),
        "--cuda-gpu-arch=sm_70".into(),
        "-D__CUDA_ARCH__=700".into(),
    ]);

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
