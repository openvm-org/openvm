use std::{fs::read, path::PathBuf, time::Instant};

use clap::{command, Parser};
use eyre::Result;
use metrics::{counter, gauge, Gauge};
use openvm_build::{build_guest_package, get_package, guest_methods, GuestOptions};
use openvm_circuit::arch::{instructions::exe::VmExe, VirtualMachine, VmConfig};
use openvm_sdk::{
    commit::commit_app_exe,
    config::AppConfig,
    keygen::{leaf_keygen, AppProvingKey},
    prover::{AppProver, LeafProver},
    StdIn,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    engine::StarkFriEngine,
    openvm_stark_backend::Chip,
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};
use tempfile::tempdir;

type F = BabyBear;
type SC = BabyBearPoseidon2Config;

#[derive(Parser, Debug)]
#[command(allow_external_subcommands = true)]
pub struct BenchmarkCli {
    /// Application level log blowup, default set by the benchmark
    #[arg(short = 'p', long, alias = "app_log_blowup")]
    pub app_log_blowup: Option<usize>,

    /// Aggregation (leaf) level log blowup, default set by the benchmark
    #[arg(short = 'g', long, alias = "agg_log_blowup")]
    pub agg_log_blowup: Option<usize>,

    /// Root level log blowup, default set by the benchmark
    #[arg(short, long, alias = "root_log_blowup")]
    pub root_log_blowup: Option<usize>,

    /// Internal level log blowup, default set by the benchmark
    #[arg(short, long, alias = "internal_log_blowup")]
    pub internal_log_blowup: Option<usize>,

    /// Max segment length for continuations
    #[arg(short, long, alias = "max_segment_length")]
    pub max_segment_length: Option<usize>,
}

fn get_programs_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    dir.push("programs");
    dir
}

pub fn build_bench_program(program_name: &str) -> Result<Elf> {
    let manifest_dir = get_programs_dir().join(program_name);
    let pkg = get_package(manifest_dir);
    let target_dir = tempdir()?;
    // Build guest with default features
    let guest_opts = GuestOptions::default().with_target_dir(target_dir.path());
    if let Err(Some(code)) = build_guest_package(&pkg, &guest_opts, None) {
        std::process::exit(code);
    }
    // Assumes the package has a single target binary
    let elf_path = guest_methods(&pkg, &target_dir, &[]).pop().unwrap();
    let data = read(elf_path)?;
    Elf::decode(&data, MEM_SIZE as u32)
}

/// 1. Generate proving key from config.
/// 2. Commit to the exe by generating cached trace for program.
/// 3. Executes runtime without metric collection and generate trace.
/// 4. Executes runtime once with full metric collection for flamegraphs (slow).
/// 5. Generate STARK proofs for each segment (segmentation is determined by `config`), with timer.
/// 6. Verify STARK proofs.
///
/// Returns the data necessary for proof aggregation.
pub fn bench_from_exe<VC>(
    bench_name: impl ToString,
    app_config: AppConfig<VC>,
    exe: impl Into<VmExe<F>>,
    input_stream: StdIn,
    bench_leaf: bool,
) -> Result<()>
where
    VC: VmConfig<F>,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    counter!("fri.log_blowup").absolute(app_config.app_fri_params.fri_params.log_blowup as u64);
    let engine = BabyBearPoseidon2Engine::new(app_config.app_fri_params.fri_params);
    let vm = VirtualMachine::new(engine, app_config.app_vm_config.clone());
    // 1. Generate proving key from config.
    let app_pk = time(gauge!("keygen_time_ms"), || {
        AppProvingKey::keygen(app_config.clone())
    });
    // 2. Commit to the exe by generating cached trace for program.
    let committed_exe = time(gauge!("commit_exe_time_ms"), || {
        commit_app_exe(app_config.app_fri_params.fri_params, exe)
    });
    // 3. Executes runtime once with full metric collection for flamegraphs (slow).
    // 4. Executes runtime again without metric collection and generate trace.
    // 5. Generate STARK proofs for each segment (segmentation is determined by `config`), with timer.
    // generate_app_proof will emit metrics for proof time of each
    let vk = app_pk.app_vm_pk.vm_pk.get_vk();
    let prover = AppProver::new(app_pk.app_vm_pk, committed_exe)
        .with_profiling()
        .with_program_name(bench_name.to_string());
    let app_proofs = prover.generate_app_proof(input_stream);
    // 6. Verify STARK proofs.
    vm.verify(&vk, app_proofs.per_segment.clone())
        .expect("Verification failed");
    if bench_leaf {
        let leaf_vm_pk = leaf_keygen(app_config.leaf_fri_params.fri_params);
        let leaf_prover = LeafProver::new(leaf_vm_pk, app_pk.leaf_committed_exe).with_profile();
        leaf_prover.generate_proof(&app_proofs);
    }
    Ok(())
}

pub fn time<F: FnOnce() -> R, R>(gauge: Gauge, f: F) -> R {
    let start = Instant::now();
    let res = f();
    gauge.set(start.elapsed().as_millis() as f64);
    res
}
