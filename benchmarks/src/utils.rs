use std::{fs::read, path::PathBuf, time::Instant};

use ax_stark_sdk::{
    ax_stark_backend::config::{Com, Domain, PcsProof, PcsProverData, StarkGenericConfig, Val},
    engine::{StarkFriEngine, VerificationDataWithFriParams},
};
use axvm_build::{build_guest_package, get_package, guest_methods, GuestOptions};
use axvm_circuit::arch::{instructions::exe::AxVmExe, VirtualMachine, VmConfig, VmExecutor};
use axvm_transpiler::{axvm_platform::memory::MEM_SIZE, elf::Elf};
use eyre::Result;
use metrics::{gauge, Gauge};
use p3_field::PrimeField32;
use tempfile::tempdir;

fn get_programs_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
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

/// 0. Transpile ELF to axVM executable.
/// 1. Executes runtime once with full metric collection for flamegraphs (slow).
/// 2. Generate proving key from config and generate committed exe.
/// 3. Executes runtime again without metric collection and generate trace.
/// 4. Generate STARK proofs for each segment (segmentation is determined by `config`), with timer.
/// 5. Verify STARK proofs.
pub fn bench_from_elf<SC, E>(
    engine: E,
    mut config: VmConfig,
    elf: Elf,
    input_stream: Vec<Vec<Val<SC>>>,
) -> Result<VerificationDataWithFriParams<SC>>
where
    SC: StarkGenericConfig,
    E: StarkFriEngine<SC>,
    Val<SC>: PrimeField32,
    SC::Pcs: Sync,
    Domain<SC>: Send + Sync,
    PcsProverData<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Challenge: Send + Sync,
    PcsProof<SC>: Send + Sync,
{
    // 0. Transpile ELF to axVM executable.
    let exe = AxVmExe::<Val<SC>>::from(elf);
    // 1. Executes runtime once with full metric collection for flamegraphs (slow).
    config.collect_metrics = true;
    let executor = VmExecutor::<Val<SC>>::new(config.clone());
    executor.execute(exe.clone(), input_stream.clone())?;
    // 2. Generate proving key from config.
    config.collect_metrics = false;
    let vm = VirtualMachine::new(engine, config);
    let pk = time(gauge!("keygen_time_ms"), || vm.keygen());
    // 3. Executes runtime again without metric collection and generate trace.
    let results = time(gauge!("trace_gen_time_ms"), || {
        vm.execute_and(exe, input_stream)
    })?;
    // 4. Generate STARK proofs for each segment (segmentation is determined by `config`), with timer.
    let proofs = vm.prove(&pk, results);
    todo!()
}

fn time<F: FnOnce() -> R, R>(gauge: Gauge, f: F) -> R {
    let start = Instant::now();
    let res = f();
    gauge.set(start.elapsed().as_millis() as f64);
    res
}
