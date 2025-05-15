use std::fs;

use clap::{arg, Parser};
use eyre::Result;
use openvm_benchmarks_utils::{build_elf_with_path, get_elf_path_with_pkg, get_programs_dir};
use openvm_build::get_package;
use tracing_subscriber::{fmt, EnvFilter};

#[derive(Parser)]
#[command(author, version, about = "Build OpenVM benchmark programs")]
struct Cli {
    /// Force rebuild even if the output ELF already exists
    #[arg(short, long)]
    force: bool,

    /// Specific program directories to build (builds all if not specified)
    #[arg(value_name = "PROGRAM")]
    programs: Vec<String>,

    /// Programs to skip
    #[arg(long, value_name = "PROGRAM")]
    skip: Vec<String>,

    /// Build profile (debug or release)
    #[arg(short, long, default_value = "release")]
    profile: String,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let filter = if cli.verbose {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::new("info")
    };
    fmt::fmt().with_env_filter(filter).init();

    let programs_dir = get_programs_dir();
    tracing::info!("Building programs from {}", programs_dir.display());

    // Collect all available program directories
    let available_programs = fs::read_dir(&programs_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();

            if path.is_dir() {
                let dir_name = path.file_name()?.to_str()?.to_string();
                Some((dir_name, path))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    // Filter programs if specific ones were requested
    let programs_to_build = if cli.programs.is_empty() {
        available_programs
    } else {
        for prog in &cli.programs {
            if !available_programs.iter().any(|(name, _)| name == prog) {
                tracing::warn!("Program '{}' not found in available programs", prog);
            }
        }

        available_programs
            .into_iter()
            .filter(|(name, _)| cli.programs.contains(name))
            .collect()
    };

    // Filter out skipped programs
    for prog in &cli.skip {
        if !programs_to_build.iter().any(|(name, _)| name == prog) {
            tracing::warn!("Program '{}' not found in programs to skip", prog);
        }
    }

    let programs_to_build = programs_to_build
        .into_iter()
        .filter(|(name, _)| !cli.skip.contains(name))
        .collect::<Vec<_>>();

    if programs_to_build.is_empty() {
        tracing::warn!("No matching programs found to build");
        return Ok(());
    }

    // Build each selected program
    for (dir_name, path) in programs_to_build {
        let pkg = get_package(&path);
        let elf_path = get_elf_path_with_pkg(&path, &pkg);

        if cli.force || !elf_path.exists() {
            tracing::info!("Building: {}", dir_name);
            build_elf_with_path(&pkg, &cli.profile, Some(&elf_path))?;
        } else {
            tracing::info!(
                "Skipping existing build: {} (use --force to rebuild)",
                dir_name
            );
        }
    }

    tracing::info!("Build complete");
    Ok(())
}
