use std::{
    collections::BTreeSet,
    env,
    path::{Path, PathBuf},
};

use walkdir::WalkDir;

pub fn is_cuda_path(path: &Path) -> bool {
    path.to_string_lossy().to_lowercase().contains("cuda")
}

pub fn has_cuda_files(path: &Path) -> bool {
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

pub fn find_cuda_include_dirs(workspace_root: &Path) -> Vec<PathBuf> {
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

pub fn find_files_with_extension(root: &Path, extension: &str) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.file_type().is_file()
                && e.path().extension().and_then(|s| s.to_str()) == Some(extension)
        })
        .map(|e| e.path().to_path_buf())
        .collect()
}

pub fn get_cuda_dep_common_include_dirs() -> Vec<PathBuf> {
    if let Some(val) = option_env!("DEP_CUDA_COMMON_INCLUDE") {
        env::split_paths(val)
            .filter(|p| !p.as_os_str().is_empty())
            .collect()
    } else {
        vec![]
    }
}
