use eyre::Result;
use openvm_benchmarks_utils::{build_and_save_elf, get_elf_path, get_programs_dir};
use openvm_build::get_package;
use std::fs;
use tracing_subscriber::{fmt, EnvFilter};

fn main() -> Result<()> {
    fmt::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let programs_dir = get_programs_dir();
    tracing::info!("Building programs from {}", programs_dir.display());

    // Iterate over all directories in programs_dir
    for entry in fs::read_dir(&programs_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            let dir_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            tracing::info!("Building: {}", dir_name);

            let pkg = get_package(&path);
            let elf_path = get_elf_path(&pkg);

            build_and_save_elf(&pkg, &elf_path, "release")?;
        }
    }

    tracing::info!("Build complete");
    Ok(())
}
