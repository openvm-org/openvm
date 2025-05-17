use eyre::Result;
use openvm_benchmarks_utils::{get_elf_path, get_programs_dir, read_elf_file};
use openvm_bigint_circuit::Int256Rv32Config;
use openvm_bigint_transpiler::Int256TranspilerExtension;
use openvm_circuit::arch::{instructions::exe::VmExe, VmExecutor};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_transpiler::{transpiler::Transpiler, FromElf};

static AVAILABLE_PROGRAMS: &[&str] = &[
    "fibonacci_recursive",
    "fibonacci_iterative",
    "quicksort",
    "bubblesort",
    "factorial_iterative_u256",
    "revm_snailtracer",
    // "pairing",
    // "keccak256",
    // "keccak256_iter",
    // "sha256",
    // "sha256_iter",
    // "revm_transfer",
];

fn main() {
    divan::main();
}

/// Run a specific OpenVM program
fn run_program(program: &str) -> Result<()> {
    let program_dir = get_programs_dir().join(program);
    let elf_path = get_elf_path(&program_dir);
    let elf = read_elf_file(&elf_path)?;

    let vm_config = Int256Rv32Config::default();

    let transpiler = Transpiler::<BabyBear>::default()
        .with_extension(Rv32ITranspilerExtension)
        .with_extension(Rv32IoTranspilerExtension)
        .with_extension(Rv32MTranspilerExtension)
        .with_extension(Int256TranspilerExtension);

    let exe = VmExe::from_elf(elf, transpiler)?;

    let executor = VmExecutor::new(vm_config);
    executor
        .execute_e1(exe, vec![])
        .expect("Failed to execute program");

    Ok(())
}

#[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=10)]
fn benchmark_execute(program: &str) {
    run_program(program).unwrap();
}
