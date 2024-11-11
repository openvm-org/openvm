use std::{fs::read, path::PathBuf, str::FromStr, time::Instant};

use anstyle::*;
use ax_stark_sdk::{
    ax_stark_backend::{
        config::{Com, Domain, PcsProof, PcsProverData, StarkGenericConfig, Val},
        p3_field::PrimeField32,
    },
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    engine::StarkFriEngine,
};
use axvm_circuit::arch::{instructions::exe::AxVmExe, ExecutorName, VirtualMachine, VmConfig};
use axvm_transpiler::{axvm_platform::memory::MEM_SIZE, elf::Elf};
use clap::Parser;
use eyre::Result;

use super::build::{build, BuildArgs};
use crate::util::write_status;

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum Input {
    FilePath(PathBuf),
    HexBytes(Vec<u8>),
}

fn is_valid_hex_string(s: &str) -> bool {
    if s.len() % 2 != 0 {
        return false;
    }
    // All hex digits with optional 0x prefix
    s.starts_with("0x") && s[2..].chars().all(|c| c.is_ascii_hexdigit())
        || s.chars().all(|c| c.is_ascii_hexdigit())
}

impl FromStr for Input {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if is_valid_hex_string(s) {
            // Remove 0x prefix if present
            let s = if s.starts_with("0x") {
                s.strip_prefix("0x").unwrap()
            } else {
                s
            };
            if s.is_empty() {
                return Ok(Input::HexBytes(Vec::new()));
            }
            if !s.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err("Invalid hex string.".to_string());
            }
            let bytes = hex::decode(s).map_err(|e| e.to_string())?;
            Ok(Input::HexBytes(bytes))
        } else if PathBuf::from(s).exists() {
            Ok(Input::FilePath(PathBuf::from(s)))
        } else {
            Err("Input must be a valid file path or hex string.".to_string())
        }
    }
}

#[derive(Parser)]
#[command(name = "prove", about = "(default) Build and prove a program")]
pub struct BenchCmd {
    #[clap(long, value_parser)]
    input: Option<Input>,

    #[clap(long, action)]
    output: Option<PathBuf>,

    #[clap(long, action)]
    profile: bool,

    #[clap(long, action)]
    verbose: bool,

    #[clap(flatten)]
    build_args: BuildArgs,
}

impl BenchCmd {
    pub fn run(&self) -> Result<()> {
        let elf_path = build(&self.build_args)?;

        if self.profile {
            setup_tracing();
        }

        let data = read(elf_path)?;
        let elf = Elf::decode(&data, MEM_SIZE as u32)?;

        // TODO: read from axiom.toml
        let app_log_blowup = 2;
        let engine = BabyBearPoseidon2Engine::new(
            FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
        );
        let config = VmConfig::rv32im().add_executor(ExecutorName::Keccak256Rv32);

        let total_proving_time_ms = bench_from_exe(engine, config, elf, vec![])?;

        let green = AnsiColor::Green.on_default().effects(Effects::BOLD);
        write_status(
            &green,
            "Finished",
            &format!("proving in {}ms", total_proving_time_ms),
        );

        Ok(())
    }
}

/// Bench without collecting metrics.
/// Performs proving keygen and then execute and proof generation.
///
/// Returns total proving time in ms.
pub fn bench_from_exe<SC, E>(
    engine: E,
    config: VmConfig,
    exe: impl Into<AxVmExe<Val<SC>>>,
    input_stream: Vec<Vec<Val<SC>>>,
) -> Result<u128>
where
    SC: StarkGenericConfig,
    E: StarkFriEngine<SC>,
    Val<SC>: PrimeField32,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    let exe = exe.into();
    // 1. Generate proving key from config.
    tracing::info!("fri.log_blowup: {}", engine.fri_params().log_blowup);
    let vm = VirtualMachine::new(engine, config);
    let pk = vm.keygen();
    // 2. Commit to the exe by generating cached trace for program.
    let committed_exe = vm.commit_exe(exe);
    // 3. Executes runtime again without metric collection and generate trace.
    let start = Instant::now();
    let results = vm.execute_and_generate_with_cached_program(committed_exe, input_stream)?;
    let execute_and_trace_gen_time_ms = start.elapsed().as_millis();
    // 4. Generate STARK proofs for each segment (segmentation is determined by `config`), with timer.
    // vm.prove will emit metrics for proof time of each segment
    let start = Instant::now();
    let proofs = vm.prove(&pk, results);
    let proving_time_ms = start.elapsed().as_millis();

    let total_proving_time_ms = execute_and_trace_gen_time_ms + proving_time_ms;

    // 6. Verify STARK proofs.
    let vk = pk.get_vk();
    vm.verify(&vk, proofs.clone()).expect("Verification failed");

    Ok(total_proving_time_ms)
}
