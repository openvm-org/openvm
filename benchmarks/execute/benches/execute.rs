use std::{path::Path, sync::OnceLock};

use divan::Bencher;
use eyre::Result;
use openvm_benchmarks_utils::{get_elf_path, get_programs_dir, read_elf_file};
use openvm_bigint_circuit::{Int256, Int256Executor, Int256Periphery};
use openvm_bigint_transpiler::Int256TranspilerExtension;
use openvm_circuit::{
    arch::{
        create_initial_state, instructions::exe::VmExe, InitFileGenerator, SystemConfig,
        VirtualMachine,
    },
    derive::VmConfig,
};
use openvm_keccak256_circuit::{Keccak256, Keccak256Executor, Keccak256Periphery};
use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sha256_circuit::{Sha256, Sha256Executor, Sha256Periphery};
use openvm_sha256_transpiler::Sha256TranspilerExtension;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::{
        default_engine, BabyBearPoseidon2Config, BabyBearPoseidon2Engine,
    },
    openvm_stark_backend::{self, p3_field::PrimeField32},
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::{transpiler::Transpiler, FromElf};
use serde::{Deserialize, Serialize};

static AVAILABLE_PROGRAMS: &[&str] = &[
    "fibonacci_recursive",
    "fibonacci_iterative",
    "quicksort",
    "bubblesort",
    "factorial_iterative_u256",
    "revm_snailtracer",
    "keccak256",
    "keccak256_iter",
    "sha256",
    "sha256_iter",
    // "revm_transfer",
    // "pairing",
];

static SHARED_WIDTHS_AND_INTERACTIONS: OnceLock<(Vec<usize>, Vec<usize>)> = OnceLock::new();

// TODO(ayush): remove from here
#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct ExecuteConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub bigint: Int256,
    #[extension]
    pub keccak: Keccak256,
    #[extension]
    pub sha256: Sha256,
}

impl Default for ExecuteConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            rv32i: Rv32I::default(),
            rv32m: Rv32M::default(),
            io: Rv32Io::default(),
            bigint: Int256::default(),
            keccak: Keccak256::default(),
            sha256: Sha256::default(),
        }
    }
}

impl InitFileGenerator for ExecuteConfig {
    fn write_to_init_file(
        &self,
        _manifest_dir: &Path,
        _init_file_name: Option<&str>,
    ) -> eyre::Result<()> {
        Ok(())
    }
}

fn main() {
    divan::main();
}

fn create_default_vm(
) -> VirtualMachine<BabyBearPoseidon2Config, BabyBearPoseidon2Engine, ExecuteConfig> {
    let vm_config = ExecuteConfig::default();
    VirtualMachine::new(default_engine(), vm_config)
}

fn create_default_transpiler() -> Transpiler<BabyBear> {
    Transpiler::<BabyBear>::default()
        .with_extension(Rv32ITranspilerExtension)
        .with_extension(Rv32IoTranspilerExtension)
        .with_extension(Rv32MTranspilerExtension)
        .with_extension(Int256TranspilerExtension)
        .with_extension(Keccak256TranspilerExtension)
        .with_extension(Sha256TranspilerExtension)
}

fn load_program_executable(program: &str) -> Result<VmExe<BabyBear>> {
    let transpiler = create_default_transpiler();
    let program_dir = get_programs_dir().join(program);
    let elf_path = get_elf_path(&program_dir);
    let elf = read_elf_file(&elf_path)?;
    Ok(VmExe::from_elf(elf, transpiler)?)
}

fn shared_widths_and_interactions() -> &'static (Vec<usize>, Vec<usize>) {
    SHARED_WIDTHS_AND_INTERACTIONS.get_or_init(|| {
        let vm = create_default_vm();
        let pk = vm.keygen();
        let vk = pk.get_vk();
        (vk.total_widths(), vk.num_interactions())
    })
}

#[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=10)]
fn benchmark_execute(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let vm = create_default_vm();
            let exe = load_program_executable(program).expect("Failed to load program executable");
            let state = create_initial_state(&vm.config().system.memory_config, &exe, vec![]);
            (vm.executor, exe, state)
        })
        .bench_values(|(executor, exe, state)| {
            executor
                .execute_e1_from_state(exe, state, None)
                .expect("Failed to execute program");
        });
}

#[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=5)]
fn benchmark_execute_metered(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let vm = create_default_vm();
            let exe = load_program_executable(program).expect("Failed to load program executable");
            let state = create_initial_state(&vm.config().system.memory_config, &exe, vec![]);

            let (widths, interactions) = shared_widths_and_interactions();
            (vm.executor, exe, state, widths, interactions)
        })
        .bench_values(|(executor, exe, state, widths, interactions)| {
            executor
                .execute_metered_from_state(exe, state, widths, interactions)
                .expect("Failed to execute program");
        });
}

// #[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=3)]
// fn benchmark_execute_e3(bencher: Bencher, program: &str) {
//     bencher
//         .with_inputs(|| {
//             let vm = create_default_vm();
//             let exe = load_program_executable(program).expect("Failed to load program executable");
//             let state = create_initial_state(&vm.config().system.memory_config, &exe, vec![]);

//             let (widths, interactions) = shared_widths_and_interactions();
//             let segments = vm
//                 .executor
//                 .execute_metered(exe.clone(), vec![], widths, interactions)
//                 .expect("Failed to execute program");

//             (vm.executor, exe, state, segments)
//         })
//         .bench_values(|(executor, exe, state, segments)| {
//             executor
//                 .execute_from_state(exe, state, &segments)
//                 .expect("Failed to execute program");
//         });
// }
