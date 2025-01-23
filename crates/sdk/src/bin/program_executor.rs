use std::path::PathBuf;

use clap::Parser;
use openvm_sdk::{
    commit::commit_app_exe,
    config::{AppConfig, SdkVmConfig},
    fs::read_app_pk_from_file,
    keygen::AppProvingKey,
    prover::vm::local::VmLocalProver,
    utils::*,
    Sdk,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Engine;

#[derive(Debug, Parser)]
struct ExecutorArgs {
    /// local path to the openvm config
    #[clap(long)]
    openvm_config_path: Option<PathBuf>,
    /// local path to the guest program dir
    #[clap(long)]
    program_dir: PathBuf,
    /// local path to the app proving key
    #[clap(long)]
    app_pk_path: PathBuf,
    /// local path that the program will write the output (segments) to
    #[clap(long)]
    output_dir: PathBuf,
}

fn main() {
    let args = ExecutorArgs::parse();

    let app_config: AppConfig<SdkVmConfig> = if let Some(path) = args.openvm_config_path {
        read_to_struct_toml(&path).unwrap()
    } else {
        default_app_config()
    };

    let app_pk: AppProvingKey<SdkVmConfig> = read_app_pk_from_file(args.app_pk_path).unwrap();

    let elf = build_program(args.program_dir, "release").unwrap();
    let transpiler = app_config.app_vm_config.transpiler();
    let exe = Sdk.transpile(elf, transpiler).unwrap();
    let committed_exe = commit_app_exe(app_pk.app_fri_params(), exe);
    let _vm: VmLocalProver<_, _, BabyBearPoseidon2Engine> =
        VmLocalProver::new(app_pk.app_vm_pk.clone(), committed_exe);
    // TODO: execute until segment
}
