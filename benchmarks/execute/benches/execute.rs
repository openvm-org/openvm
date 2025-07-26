use std::{path::Path, sync::OnceLock};

use divan::Bencher;
use eyre::Result;
use openvm_algebra_circuit::{
    AlgebraCpuProverExt, Fp2Extension, Fp2ExtensionExecutor, ModularExtension,
    ModularExtensionExecutor,
};
use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use openvm_benchmarks_utils::{get_elf_path, get_programs_dir, read_elf_file};
use openvm_bigint_circuit::{Int256, Int256CpuProverExt, Int256Executor};
use openvm_bigint_transpiler::Int256TranspilerExtension;
use openvm_circuit::{
    arch::{
        execution_mode::{e1::E1Ctx, metered::MeteredCtx},
        instructions::exe::VmExe,
        interpreter::InterpretedInstance,
        *,
    },
    derive::VmConfig,
    system::*,
};
use openvm_ecc_circuit::{EccCpuProverExt, WeierstrassExtension, WeierstrassExtensionExecutor};
use openvm_ecc_transpiler::EccTranspilerExtension;
use openvm_keccak256_circuit::{Keccak256, Keccak256CpuProverExt, Keccak256Executor};
use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_pairing_circuit::{
    PairingCurve, PairingExtension, PairingExtensionExecutor, PairingProverExt,
};
use openvm_pairing_guest::bn254::BN254_COMPLEX_STRUCT_NAME;
use openvm_pairing_transpiler::PairingTranspilerExtension;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32ImCpuProverExt, Rv32Io, Rv32IoExecutor, Rv32M, Rv32MExecutor,
};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sha256_circuit::{Sha256, Sha256Executor, Sha2CpuProverExt};
use openvm_sha256_transpiler::Sha256TranspilerExtension;
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::{StarkEngine, StarkFriEngine},
    openvm_stark_backend::{
        self,
        config::{StarkGenericConfig, Val},
        p3_field::PrimeField32,
        prover::cpu::{CpuBackend, CpuDevice},
    },
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

static METERED_CTX: OnceLock<(MeteredCtx, Vec<usize>)> = OnceLock::new();

#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct ExecuteConfig {
    #[config(executor = "SystemExecutor<F>")]
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
    #[extension(generics = true)]
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

pub struct ExecuteBuilder;
impl<E, SC> VmBuilder<E> for ExecuteBuilder
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
{
    type VmConfig = ExecuteConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &ExecuteConfig,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemCpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.rv32i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.rv32m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &Int256CpuProverExt,
            &config.bigint,
            inventory,
        )?;
        VmProverExtension::<E, _, _>::extend_prover(
            &Keccak256CpuProverExt,
            &config.keccak,
            inventory,
        )?;
        VmProverExtension::<E, _, _>::extend_prover(&Sha2CpuProverExt, &config.sha256, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &AlgebraCpuProverExt,
            &config.modular,
            inventory,
        )?;
        VmProverExtension::<E, _, _>::extend_prover(&AlgebraCpuProverExt, &config.fp2, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &EccCpuProverExt,
            &config.weierstrass,
            inventory,
        )?;
        VmProverExtension::<E, _, _>::extend_prover(&PairingProverExt, &config.pairing, inventory)?;
        Ok(chip_complex)
    }
}

fn main() {
    divan::main();
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

fn metering_setup() -> &'static (MeteredCtx, Vec<usize>) {
    METERED_CTX.get_or_init(|| {
        let config = ExecuteConfig::default();
        let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
        let (vm, _) = VirtualMachine::new_with_keygen(engine, ExecuteBuilder, config).unwrap();
        let ctx = vm.build_metered_ctx();
        let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
        (ctx, executor_idx_to_air_idx)
    })
}

#[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=10)]
fn benchmark_execute(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let vm_config = ExecuteConfig::default();
            let exe = load_program_executable(program).expect("Failed to load program executable");
            let interpreter = InterpretedInstance::new(vm_config, exe).unwrap();
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

            let (ctx, executor_idx_to_air_idx) = metering_setup();
            let interpreter = InterpretedInstance::new(vm_config, exe).unwrap();

            (interpreter, vec![], ctx.clone(), executor_idx_to_air_idx)
        })
        .bench_values(|(interpreter, input, ctx, executor_idx_to_air_idx)| {
            interpreter
                .execute_e2(ctx, input, executor_idx_to_air_idx)
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
