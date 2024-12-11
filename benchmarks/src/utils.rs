use std::{fs::read, path::PathBuf, time::Instant};

use ax_stark_sdk::{
    ax_stark_backend::{engine::VerificationData, Chip},
    config::baby_bear_poseidon2::BabyBearPoseidon2Config,
    engine::{StarkFriEngine, VerificationDataWithFriParams},
    p3_baby_bear::BabyBear,
};
use axvm_build::{build_guest_package, get_package, guest_methods, GuestOptions};
use axvm_circuit::arch::{instructions::exe::AxVmExe, VirtualMachine, VmConfig};
use axvm_sdk::{
    commit::commit_app_exe, config::AppConfig, keygen::AppProvingKey, prover::AppProver, StdIn,
};
use axvm_transpiler::{axvm_platform::memory::MEM_SIZE, elf::Elf};
use clap::{command, Parser};
use eyre::Result;
use metrics::{counter, gauge, Gauge};
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
pub fn bench_from_exe<E, VC>(
    engine: E,
    config: VC,
    exe: impl Into<AxVmExe<F>>,
    input_stream: StdIn,
) -> Result<Vec<VerificationDataWithFriParams<SC>>>
where
    E: StarkFriEngine<SC>,
    VC: VmConfig<F>,
    VC::Executor: Chip<SC>,
    VC::Periphery: Chip<SC>,
{
    counter!("fri.log_blowup").absolute(engine.fri_params().log_blowup as u64);
    let app_config = AppConfig {
        app_vm_config: config.clone(),
        app_fri_params: engine.fri_params(),
        // leaf_fri_params/compiler_options don't matter for this benchmark.
        leaf_fri_params: engine.fri_params().into(),
        compiler_options: Default::default(),
    };
    let vm = VirtualMachine::new(engine, config);
    // 1. Generate proving key from config.
    let app_pk = time(gauge!("keygen_time_ms"), || {
        AppProvingKey::keygen(app_config.clone())
    });
    // 2. Commit to the exe by generating cached trace for program.
    let committed_exe = time(gauge!("commit_exe_time_ms"), || {
        commit_app_exe(app_config.app_fri_params, exe)
    });
    // 3. Executes runtime again without metric collection and generate trace.
    time(gauge!("execute_and_trace_gen_time_ms"), || {
        vm.execute_and_generate_with_cached_program(committed_exe.clone(), input_stream.clone())
    })?;
    // 4. Executes runtime once with full metric collection for flamegraphs (slow).
    // 5. Generate STARK proofs for each segment (segmentation is determined by `config`), with timer.
    // generate_app_proof will emit metrics for proof time of each
    let vk = app_pk.app_vm_pk.vm_pk.get_vk();
    let mut prover = AppProver::new(app_pk.app_vm_pk, committed_exe);
    prover.profile = true;
    let proofs = prover.generate_app_proof(input_stream).per_segment;
    // 6. Verify STARK proofs.
    vm.verify(&vk, proofs.clone()).expect("Verification failed");
    let vdata = proofs
        .into_iter()
        .map(|proof| VerificationDataWithFriParams {
            data: VerificationData {
                vk: vk.clone(),
                proof,
            },
            fri_params: vm.engine.fri_params(),
        })
        .collect();
    Ok(vdata)
}

fn time<F: FnOnce() -> R, R>(gauge: Gauge, f: F) -> R {
    let start = Instant::now();
    let res = f();
    gauge.set(start.elapsed().as_millis() as f64);
    res
}
