use std::{fs, sync::Arc};

use eyre::Result;
use openvm_benchmarks_utils::{get_elf_path, get_fixtures_dir, get_programs_dir, read_elf_file};
use openvm_circuit::arch::{instructions::exe::VmExe, VmCircuitConfig};
use openvm_continuations::verifier::common::types::VmVerifierPvs;
use openvm_native_circuit::NativeConfig;
use openvm_sdk::{
    commit::commit_app_exe,
    config::{
        AppConfig, AppFriParams, LeafFriParams, SdkVmConfig, SdkVmCpuBuilder,
        DEFAULT_APP_LOG_BLOWUP, DEFAULT_LEAF_LOG_BLOWUP, SBOX_SIZE,
    },
    Sdk, StdIn,
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::StarkFriEngine,
};
use openvm_transpiler::FromElf;
use tracing_subscriber::{fmt, EnvFilter};

const PROGRAM: &str = "kitchen-sink";

fn main() -> Result<()> {
    // Set up logging
    fmt::fmt().with_env_filter(EnvFilter::new("info")).init();

    let program_dir = get_programs_dir().join(PROGRAM);

    tracing::info!("Loading VM config");
    let config_path = program_dir.join("openvm.toml");
    let config_content = fs::read_to_string(&config_path)?;
    let vm_config = SdkVmConfig::from_toml(&config_content)?.app_vm_config;

    tracing::info!("Preparing ELF");
    let elf_path = get_elf_path(&program_dir);
    let elf = read_elf_file(&elf_path)?;

    let exe = VmExe::from_elf(elf, vm_config.transpiler())?;

    let sdk = Sdk::new();

    // Create app config with default parameters
    let app_config = AppConfig {
        app_fri_params: AppFriParams {
            fri_params: FriParameters::standard_with_100_bits_conjectured_security(
                DEFAULT_APP_LOG_BLOWUP,
            ),
        },
        leaf_fri_params: LeafFriParams {
            fri_params: FriParameters::standard_with_100_bits_conjectured_security(
                DEFAULT_LEAF_LOG_BLOWUP,
            ),
        },
        app_vm_config: vm_config,
        compiler_options: Default::default(),
    };

    tracing::info!("Generating app proving key");
    let app_pk = Arc::new(sdk.app_keygen(app_config.clone())?);
    let app_committed_exe = commit_app_exe(app_pk.app_fri_params(), exe);

    tracing::info!("Generating app proof");
    let app_proof = sdk.generate_app_proof(
        SdkVmCpuBuilder,
        app_pk.clone(),
        app_committed_exe,
        StdIn::default(),
    )?;

    tracing::info!("Generating leaf proving key");
    // Generate leaf VM proving key using the circuit keygen approach
    let leaf_vm_config = NativeConfig::aggregation(
        VmVerifierPvs::<u8>::width(),
        SBOX_SIZE.min(
            app_config
                .leaf_fri_params
                .fri_params
                .max_constraint_degree(),
        ),
    );
    let circuit = leaf_vm_config.create_airs()?;
    let engine = BabyBearPoseidon2Engine::new(app_config.leaf_fri_params.fri_params);
    let pk = circuit.keygen(&engine);

    tracing::info!("Saving keys and proof to files");
    // Create fixtures directory if it doesn't exist
    let fixtures_dir = get_fixtures_dir();
    fs::create_dir_all(&fixtures_dir)?;

    // Serialize and write to files in fixtures directory
    let leaf_exe_bytes = bitcode::serialize(&app_pk.leaf_committed_exe.exe)?;
    fs::write(
        fixtures_dir.join(&format!("{}.leaf.exe", PROGRAM)),
        leaf_exe_bytes,
    )?;

    let leaf_pk_bytes = bitcode::serialize(&pk)?;
    fs::write(
        fixtures_dir.join(&format!("{}.leaf.pk", PROGRAM)),
        leaf_pk_bytes,
    )?;

    let app_proof_bytes = bitcode::serialize(&app_proof)?;
    fs::write(
        fixtures_dir.join(&format!("{}.app.proof", PROGRAM)),
        app_proof_bytes,
    )?;

    tracing::info!(
        "Generated and saved {name}.leaf.committed.exe, {name}.leaf.pk, and {name}.app.proof",
        name = PROGRAM
    );

    Ok(())
}
