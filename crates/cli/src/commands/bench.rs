use std::{path::PathBuf, time::Instant};

use anstyle::*;
use clap::Parser;
use eyre::Result;
use openvm_circuit::{
    arch::{
        hasher::poseidon2::vm_poseidon2_hasher, instructions::exe::VmExe, VirtualMachine, VmConfig,
    },
    system::memory::tree::public_values::UserPublicValuesProof,
};
use openvm_keccak256_circuit::Keccak256Rv32Config;
use openvm_sdk::fs::read_exe_from_file;
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    engine::StarkFriEngine,
    openvm_stark_backend::{
        config::{Com, StarkGenericConfig, Val},
        p3_field::PrimeField32,
        Chip,
    },
};

use super::build::{build, BuildArgs};
use crate::util::{classical_exe_path, write_status, Input};

#[derive(Clone, Parser)]
#[command(name = "bench", about = "(default) Build and prove a program")]
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
        if self.profile {
            setup_tracing();
        }
        let elf_path = build(&self.build_args)?.unwrap();
        let exe_path = classical_exe_path(&elf_path);
        let exe = read_exe_from_file(&exe_path)?;

        let app_log_blowup = 2;
        let engine = BabyBearPoseidon2Engine::new(
            FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
        );
        let config = Keccak256Rv32Config::default();

        let total_proving_time_ms = bench_from_exe(engine, config, exe, vec![])?;

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
pub fn bench_from_exe<SC, E, VC>(
    engine: E,
    config: VC,
    exe: impl Into<VmExe<Val<SC>>>,
    input_stream: Vec<Vec<Val<SC>>>,
) -> Result<u128>
where
    SC: StarkGenericConfig,
    E: StarkFriEngine<SC>,
    Val<SC>: PrimeField32,
    VC: VmConfig<Val<SC>>,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
    Com<SC>: Into<[Val<SC>; 8]>,
{
    let exe = exe.into();
    // 1. Generate proving key from config.
    tracing::info!("fri.log_blowup: {}", engine.fri_params().log_blowup);
    let system_config = config.system().clone();
    let vm = VirtualMachine::<SC, E, VC>::new(engine, config);
    let pk = vm.keygen();
    // 2. Commit to the exe by generating cached trace for program.
    let committed_exe = vm.commit_exe(exe);
    // 3. Executes runtime again without metric collection and generate trace.
    let start = Instant::now();
    let results = vm.execute_and_generate_with_cached_program(committed_exe, input_stream)?;
    let user_pv_proof = UserPublicValuesProof::compute(
        system_config.memory_config.memory_dimensions(),
        system_config.num_public_values,
        &vm_poseidon2_hasher(),
        results.final_memory.as_ref().unwrap(),
    );
    let execute_and_trace_gen_time_ms = start.elapsed().as_millis();
    // 4. Generate STARK proofs for each segment (segmentation is determined by `config`), with timer.
    // vm.prove will emit metrics for proof time of each segment
    let start = Instant::now();
    let proofs = vm.prove(&pk, results);
    let proving_time_ms = start.elapsed().as_millis();

    let total_proving_time_ms = execute_and_trace_gen_time_ms + proving_time_ms;

    // 6. Verify STARK proofs.
    let vk = pk.get_vk();
    vm.verify(&vk, proofs.clone(), Some(&user_pv_proof))
        .expect("Verification failed");

    Ok(total_proving_time_ms)
}
