use std::{path::Path, sync::OnceLock};

use divan::Bencher;
use eyre::Result;
use openvm_algebra_circuit::{
    Fp2Extension, Fp2ExtensionExecutor, Fp2ExtensionPeriphery, ModularExtension,
    ModularExtensionExecutor, ModularExtensionPeriphery,
};
use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use openvm_benchmarks_utils::{get_elf_path, get_programs_dir, read_elf_file};
use openvm_bigint_circuit::{Int256, Int256Executor, Int256Periphery};
use openvm_bigint_transpiler::Int256TranspilerExtension;
use openvm_circuit::{
    arch::{
        execution_mode::{
            e1::E1Ctx,
            metered::{ctx::DEFAULT_PAGE_BITS, MeteredCtx},
        },
        instructions::exe::VmExe,
        interpreter::InterpretedInstance,
        InitFileGenerator, SystemConfig, VirtualMachine, VmChipComplex, VmConfig,
    },
    derive::VmConfig,
};
use openvm_ecc_circuit::{
    WeierstrassExtension, WeierstrassExtensionExecutor, WeierstrassExtensionPeriphery,
};
use openvm_ecc_transpiler::EccTranspilerExtension;
use openvm_keccak256_circuit::{Keccak256, Keccak256Executor, Keccak256Periphery};
use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_pairing_circuit::{
    PairingCurve, PairingExtension, PairingExtensionExecutor, PairingExtensionPeriphery,
};
use openvm_pairing_guest::bn254::BN254_COMPLEX_STRUCT_NAME;
use openvm_pairing_transpiler::PairingTranspilerExtension;
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
    "revm_transfer",
    "pairing",
];

static SHARED_INTERACTIONS: OnceLock<Vec<usize>> = OnceLock::new();

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
    #[extension]
    pub modular: ModularExtension,
    #[extension]
    pub fp2: Fp2Extension,
    #[extension]
    pub weierstrass: WeierstrassExtension,
    #[extension]
    pub pairing: PairingExtension,
}

impl Default for ExecuteConfig {
    fn default() -> Self {
        let bn_config = PairingCurve::Bn254.curve_config();
        Self {
            system: SystemConfig::default().with_continuations(),
            rv32i: Rv32I,
            rv32m: Rv32M::default(),
            io: Rv32Io,
            bigint: Int256::default(),
            keccak: Keccak256,
            sha256: Sha256,
            modular: ModularExtension::new(vec![
                bn_config.modulus.clone(),
                bn_config.scalar.clone(),
            ]),
            fp2: Fp2Extension::new(vec![(
                BN254_COMPLEX_STRUCT_NAME.to_string(),
                bn_config.modulus.clone(),
            )]),
            weierstrass: WeierstrassExtension::new(vec![bn_config.clone()]),
            pairing: PairingExtension::new(vec![PairingCurve::Bn254]),
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
        .with_extension(ModularTranspilerExtension)
        .with_extension(Fp2TranspilerExtension)
        .with_extension(EccTranspilerExtension)
        .with_extension(PairingTranspilerExtension)
}

fn load_program_executable(program: &str) -> Result<VmExe<BabyBear>> {
    let transpiler = create_default_transpiler();
    let program_dir = get_programs_dir().join(program);
    let elf_path = get_elf_path(&program_dir);
    let elf = read_elf_file(&elf_path)?;
    Ok(VmExe::from_elf(elf, transpiler)?)
}

fn shared_interactions() -> &'static Vec<usize> {
    SHARED_INTERACTIONS.get_or_init(|| {
        let vm = create_default_vm();
        let pk = vm.keygen();
        let vk = pk.get_vk();
        vk.num_interactions()
    })
}

#[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=10)]
fn benchmark_execute(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let vm_config = ExecuteConfig::default();
            let exe = load_program_executable(program).expect("Failed to load program executable");
            let interpreter = InterpretedInstance::new(vm_config, exe);
            (interpreter, vec![])
        })
        .bench_values(|(interpreter, input)| {
            interpreter
                .execute(E1Ctx::new(None), input)
                .expect("Failed to execute program in interpreted mode");
        });
}

#[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=5)]
fn benchmark_execute_metered(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let vm_config = ExecuteConfig::default();
            let exe = load_program_executable(program).expect("Failed to load program executable");

            let chip_complex: VmChipComplex<BabyBear, _, _> =
                vm_config.create_chip_complex().unwrap();
            let interactions = shared_interactions();
            let segmentation_strategy =
                &<ExecuteConfig as VmConfig<BabyBear>>::system(&vm_config).segmentation_strategy;

            let ctx: MeteredCtx<DEFAULT_PAGE_BITS> =
                MeteredCtx::new(&chip_complex, interactions.to_vec())
                    .with_max_trace_height(segmentation_strategy.max_trace_height() as u32)
                    .with_max_cells(segmentation_strategy.max_cells());
            let interpreter = InterpretedInstance::new(vm_config, exe);

            (interpreter, vec![], ctx)
        })
        .bench_values(|(interpreter, input, ctx)| {
            interpreter
                .execute_e2(ctx, input)
                .expect("Failed to execute program");
        });
}

// #[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=3)]
// fn benchmark_execute_e3(bencher: Bencher, program: &str) {
//     bencher
//         .with_inputs(|| {
//             let vm = create_default_vm();
//             let exe = load_program_executable(program).expect("Failed to load program
// executable");             let state = create_initial_state(&vm.config().system.memory_config,
// &exe, vec![], 0);

//             let (widths, interactions) = shared_widths_and_interactions();
//             let segments = vm
//                 .executor
//                 .execute_metered(exe.clone(), vec![], interactions)
//                 .expect("Failed to execute program");

//             (vm.executor, exe, state, segments)
//         })
//         .bench_values(|(executor, exe, state, segments)| {
//             executor
//                 .execute_from_state(exe, state, &segments)
//                 .expect("Failed to execute program");
//         });
// }
