use std::{fs::read, path::PathBuf, time::Instant};

use ax_stark_sdk::{
    ax_stark_backend::{
        config::{StarkGenericConfig, Val},
        engine::VerificationData,
    },
    engine::{StarkFriEngine, VerificationDataWithFriParams},
};
use axvm_build::{build_guest_package, get_package, guest_methods, GuestOptions};
use axvm_circuit::arch::{instructions::exe::AxVmExe, VirtualMachine, VmConfig, VmExecutor};
use axvm_transpiler::{axvm_platform::memory::MEM_SIZE, elf::Elf};
use clap::{command, Parser};
use eyre::Result;
use metrics::{counter, gauge, Gauge};
use p3_field::PrimeField32;
use tempfile::tempdir;

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
    let guest_opts = GuestOptions::default().into();
    build_guest_package(&pkg, &target_dir, &guest_opts, None);
    // Assumes the package has a single target binary
    let elf_path = guest_methods(&pkg, &target_dir, &[]).pop().unwrap();
    let data = read(elf_path)?;
    Elf::decode(&data, MEM_SIZE as u32)
}

/// 1. Executes runtime once with full metric collection for flamegraphs (slow).
/// 2. Generate proving key from config.
/// 3. Commit to the exe by generating cached trace for program.
/// 4. Executes runtime again without metric collection and generate trace.
/// 5. Generate STARK proofs for each segment (segmentation is determined by `config`), with timer.
/// 6. Verify STARK proofs.
///
/// Returns the data necessary for proof aggregation.
pub fn bench_from_exe<SC, E>(
    engine: E,
    mut config: VmConfig,
    exe: impl Into<AxVmExe<Val<SC>>>,
    input_stream: Vec<Vec<Val<SC>>>,
) -> Result<Vec<VerificationDataWithFriParams<SC>>>
where
    SC: StarkGenericConfig,
    E: StarkFriEngine<SC>,
    Val<SC>: PrimeField32,
{
    let exe = exe.into();
    // 1. Executes runtime once with full metric collection for flamegraphs (slow).
    config.collect_metrics = true;
    let executor = VmExecutor::<Val<SC>>::new(config.clone());
    tracing::info_span!("execute_with_metrics", collect_metrics = true)
        .in_scope(|| executor.execute(exe.clone(), input_stream.clone()))?;
    // 2. Generate proving key from config.
    config.collect_metrics = false;
    counter!("fri.log_blowup").absolute(engine.fri_params().log_blowup as u64);
    let vm = VirtualMachine::new(engine, config);
    let pk = time(gauge!("keygen_time_ms"), || vm.keygen());
    // 3. Commit to the exe by generating cached trace for program.
    let committed_exe = time(gauge!("commit_exe_time_ms"), || vm.commit_exe(exe));
    // 4. Executes runtime again without metric collection and generate trace.
    let results = time(gauge!("execute_and_trace_gen_time_ms"), || {
        vm.execute_and_generate_with_cached_program(committed_exe, input_stream)
    })?;
    // 5. Generate STARK proofs for each segment (segmentation is determined by `config`), with timer.
    // vm.prove will emit metrics for proof time of each segment
    let proofs = vm.prove(&pk, results);
    // 6. Verify STARK proofs.
    let vk = pk.get_vk();
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
