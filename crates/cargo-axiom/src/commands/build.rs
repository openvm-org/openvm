use std::{
    fs::{read, File},
    io::Write,
    path::PathBuf,
};

use axvm_build::{
    find_unique_executable, get_dir_with_profile, get_target_dir, GuestOptions, TargetFilter,
};
use axvm_rv32im_transpiler::{Rv32ITranspilerExtension, Rv32MTranspilerExtension};
use axvm_sdk::Sdk;
use axvm_transpiler::{
    axvm_platform::{bincode, memory::MEM_SIZE},
    elf::Elf,
    transpiler::Transpiler,
};
use clap::Parser;
use eyre::Result;

#[derive(Parser)]
#[command(name = "build", about = "Compile an axVM program")]
pub struct BuildCmd {
    #[clap(flatten)]
    build_args: BuildArgs,
}

impl BuildCmd {
    pub fn run(&self) -> Result<()> {
        build(&self.build_args)?;
        Ok(())
    }
}

#[derive(Parser)]
pub struct BuildArgs {
    /// Location of the directory containing the Cargo.toml for the guest code.
    ///
    /// This path is relative to the current directory.
    #[arg(long)]
    pub manifest_dir: Option<PathBuf>,

    /// Feature flags passed to cargo.
    #[arg(long, value_delimiter = ',')]
    pub features: Vec<String>,

    #[clap(flatten)]
    pub bin_type_filter: BinTypeFilter,

    /// Target name substring filter
    #[arg(long)]
    pub name: Option<String>,

    /// Transpile the program after building
    #[arg(long, default_value = "false")]
    pub do_not_transpile: bool,

    /// Output path for the transpiled program (default: <ELF base path>.axvmexe)
    #[arg(long)]
    pub transpile_path: Option<PathBuf>,

    /// Build profile
    #[arg(long, default_value = "release")]
    pub profile: String,
}

#[derive(clap::Args)]
#[group(required = false, multiple = false)]
pub struct BinTypeFilter {
    /// Specify that the target should be a binary kind
    #[arg(long)]
    pub bin: bool,

    /// Specify that the target should be an example kind
    #[arg(long)]
    pub example: bool,
}

// Returns elf_path for now
pub(crate) fn build(build_args: &BuildArgs) -> Result<PathBuf> {
    println!("[axiom] Building the package...");
    let target_filter = TargetFilter {
        name_substr: build_args.name.clone(),
        kind: if build_args.bin_type_filter.bin {
            Some("bin".to_string())
        } else if build_args.bin_type_filter.example {
            Some("example".to_string())
        } else {
            None
        },
    };
    let pkg_dir = build_args
        .manifest_dir
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap());
    let guest_options = GuestOptions {
        features: build_args.features.clone(),
        ..Default::default()
    };
    let _elf = Sdk.build(guest_options, &pkg_dir, &target_filter)?;

    let target_dir = get_dir_with_profile(get_target_dir(&pkg_dir), &build_args.profile);
    let elf_path = find_unique_executable(&pkg_dir, &target_dir, &target_filter)?;
    if !build_args.do_not_transpile {
        println!("[axiom] Transpiling the package...");
        let output_path = build_args
            .transpile_path
            .clone()
            .unwrap_or_else(|| elf_path.with_extension("axvmexe"));
        transpile(elf_path.clone(), output_path.clone())?;
        println!(
            "[axiom] Successfully transpiled to {}",
            output_path.display()
        );
        Ok(output_path)
    } else {
        println!(
            "[axiom] Successfully built the package: {}",
            elf_path.display()
        );
        Ok(elf_path)
    }
}

fn transpile(elf_path: PathBuf, output_path: PathBuf) -> Result<()> {
    let data = read(elf_path.clone())?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let exe = Sdk.transpile(
        elf,
        Transpiler::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension),
    )?;
    let data = bincode::serde::encode_to_vec(&exe, bincode::config::standard())?;
    File::create(output_path.clone())?.write_all(&data)?;
    eprintln!("Successfully transpiled to {}", output_path.display());
    Ok(())
}
