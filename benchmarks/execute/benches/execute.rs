use std::{
    collections::{HashMap, HashSet},
    io,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
};

use divan::Bencher;
use eyre::Result;
use openvm_algebra_circuit::{
    AlgebraCpuProverExt, Fp2Extension, Fp2ExtensionExecutor, ModularExtension,
    ModularExtensionExecutor,
};
use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use openvm_benchmarks_utils::{get_programs_dir, read_elf_file};
use openvm_bigint_circuit::{Int256, Int256CpuProverExt, Int256Executor};
use openvm_bigint_transpiler::Int256TranspilerExtension;
#[cfg(not(feature = "rvr"))]
use openvm_circuit::arch::execution_mode::ExecutionCtx;
#[cfg(feature = "rvr")]
use openvm_circuit::arch::rvr::{RvrMeteredCostInstance, RvrMeteredInstance, RvrPureInstance};
use openvm_circuit::{
    arch::{
        execution_mode::{MeteredCostCtx, MeteredCtx},
        instructions::exe::VmExe,
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
use openvm_riscv_circuit::{
    Rv64I, Rv64IExecutor, Rv64ImCpuProverExt, Rv64Io, Rv64IoExecutor, Rv64M, Rv64MExecutor,
};
use openvm_riscv_transpiler::{
    Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
};
use openvm_sha2_circuit::{Sha2, Sha2CpuProverExt, Sha2Executor};
use openvm_sha2_transpiler::Sha2TranspilerExtension;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine,
    openvm_cpu_backend::{CpuBackend, CpuDevice},
    openvm_stark_backend::{
        self, keygen::types::MultiStarkProvingKey, prover::DeviceDataTransporter, EngineDeviceCtx,
        StarkEngine, StarkProtocolConfig, SystemParams, Val,
    },
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::{transpiler::Transpiler, FromElf};
use serde::{Deserialize, Serialize};

const APP_PROGRAMS: &[&str] = &[
    "fibonacci_recursive",
    "fibonacci_iterative",
    "quicksort",
    "bubblesort",
    "revm_snailtracer",
    "keccak256",
    "keccak256_iter",
    "sha256",
    "sha256_iter",
    "revm_transfer",
    "pairing",
];

type SC = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
type Engine = BabyBearPoseidon2CpuEngine;

static VM_PROVING_KEY: OnceLock<MultiStarkProvingKey<SC>> = OnceLock::new();
static METERED_COST_CTX: OnceLock<(MeteredCostCtx, Vec<usize>)> = OnceLock::new();
static EXECUTOR: OnceLock<VmExecutor<BabyBear, ExecuteConfig>> = OnceLock::new();
static SUCCESSFUL_EXECUTIONS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
type Cache<T> = OnceLock<Mutex<HashMap<String, Arc<T>>>>;

fn report_program_success(mode: &str, program: &str) {
    let successes = SUCCESSFUL_EXECUTIONS.get_or_init(|| Mutex::new(HashSet::new()));
    let mut successes = successes
        .lock()
        .expect("Failed to access successful execution log");
    let key = format!("{mode}:{program}");
    if successes.insert(key) {
        println!("Succeeded {mode} execution for program `{program}`");
    }
}

#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct ExecuteConfig {
    #[config(executor = "SystemExecutor")]
    pub system: SystemConfig,
    #[extension]
    pub rv64i: Rv64I,
    #[extension]
    pub rv64m: Rv64M,
    #[extension]
    pub io: Rv64Io,
    #[extension]
    pub bigint: Int256,
    #[extension]
    pub keccak: Keccak256,
    #[extension]
    pub sha2: Sha2,
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
            system: SystemConfig::default(),
            rv64i: Rv64I,
            rv64m: Rv64M::default(),
            io: Rv64Io,
            bigint: Int256::default(),
            keccak: Keccak256,
            sha2: Sha2,
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
    ) -> io::Result<()> {
        Ok(())
    }
}

pub struct ExecuteBuilder;
impl<E, SC> VmBuilder<E> for ExecuteBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = ExecuteConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &ExecuteConfig,
        circuit: AirInventory<SC>,
        device_ctx: &EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemCpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, &config.rv64i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, &config.rv64m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImCpuProverExt, &config.io, inventory)?;
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
        VmProverExtension::<E, _, _>::extend_prover(&Sha2CpuProverExt, &config.sha2, inventory)?;
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
        .with_extension(Rv64ITranspilerExtension)
        .with_extension(Rv64IoTranspilerExtension)
        .with_extension(Rv64MTranspilerExtension)
        .with_extension(Int256TranspilerExtension)
        .with_extension(Keccak256TranspilerExtension)
        .with_extension(Sha2TranspilerExtension)
        .with_extension(ModularTranspilerExtension)
        .with_extension(Fp2TranspilerExtension)
        .with_extension(EccTranspilerExtension)
        .with_extension(PairingTranspilerExtension)
}

fn load_program_executable(program: &str) -> Result<VmExe<BabyBear>> {
    let transpiler = create_default_transpiler();
    let program_dir = get_programs_dir().join(program);
    let elf_path = openvm_benchmarks_utils::get_elf_path(&program_dir);
    let elf = read_elf_file(&elf_path)?;
    Ok(VmExe::from_elf(elf, transpiler)?)
}

fn vm_proving_key() -> &'static MultiStarkProvingKey<SC> {
    VM_PROVING_KEY.get_or_init(|| {
        let config = ExecuteConfig::default();
        let engine = Engine::new(SystemParams::new_for_testing(21));
        let (_vm, pk) = VirtualMachine::new_with_keygen(engine, ExecuteBuilder, config).unwrap();
        pk
    })
}

fn metered_cost_setup() -> &'static (MeteredCostCtx, Vec<usize>) {
    METERED_COST_CTX.get_or_init(|| {
        let config = ExecuteConfig::default();
        let engine = Engine::new(SystemParams::new_for_testing(21));
        let pk = vm_proving_key();
        let d_pk = engine.device().transport_pk_to_device(pk);
        let vm = VirtualMachine::new(engine, ExecuteBuilder, config, d_pk).unwrap();
        let ctx = vm.build_metered_cost_ctx();
        let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
        (ctx, executor_idx_to_air_idx)
    })
}

fn executor() -> &'static VmExecutor<BabyBear, ExecuteConfig> {
    EXECUTOR.get_or_init(|| {
        let vm_config = ExecuteConfig::default();
        VmExecutor::<BabyBear, _>::new(vm_config).unwrap()
    })
}

fn build_metered_ctx_for(exe: &VmExe<BabyBear>) -> (MeteredCtx, Vec<usize>) {
    let config = ExecuteConfig::default();
    let engine = Engine::new(SystemParams::new_for_testing(21));
    let pk = vm_proving_key();
    let d_pk = engine.device().transport_pk_to_device(pk);
    let vm = VirtualMachine::new(engine, ExecuteBuilder, config, d_pk)
        .expect("Failed to create VM for metered setup");
    let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
    let ctx = vm.build_metered_ctx(exe);
    (ctx, executor_idx_to_air_idx)
}

struct PureExecution;
struct MeteredExecution;
struct MeteredCostExecution;

trait BenchExecutor {
    type Instance: Send + Sync + 'static;

    fn execution_mode() -> &'static str;
    fn cache() -> &'static Cache<Self::Instance>;
    fn build_instance(exe: &VmExe<BabyBear>) -> Self::Instance;
    fn run_execution(instance: &Self::Instance, input: Vec<Vec<u8>>) -> Result<(), ExecutionError>;

    fn get_cached_instance(program: &str) -> Arc<Self::Instance> {
        let cache = Self::cache().get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache.lock().unwrap();
        cache
            .entry(program.to_string())
            .or_insert_with(|| {
                let exe = load_program_executable(program).unwrap_or_else(|err| {
                    panic!("Failed to load executable for program {program}: {err:?}")
                });
                Arc::new(Self::build_instance(&exe))
            })
            .clone()
    }

    fn unwrap_instance<T>(result: Result<T, StaticProgramError>) -> T {
        result.unwrap_or_else(|err| {
            panic!(
                "Failed to create {} instance: {err}",
                Self::execution_mode()
            )
        })
    }

    fn benchmark(bencher: Bencher, program: &str) {
        let instance = Self::get_cached_instance(program);
        bencher
            .with_inputs(Vec::<Vec<u8>>::new)
            .bench_values(|input| match Self::run_execution(&instance, input) {
                Ok(()) => report_program_success(Self::execution_mode(), program),
                Err(err) => panic!(
                    "Failed {} execution for program {program}: {err:?}",
                    Self::execution_mode()
                ),
            });
    }
}

impl BenchExecutor for PureExecution {
    #[cfg(feature = "aot")]
    type Instance = AotInstance<'static, ExecutionCtx>;
    #[cfg(feature = "rvr")]
    type Instance = RvrPureInstance<'static, BabyBear>;
    #[cfg(all(not(feature = "aot"), not(feature = "rvr")))]
    type Instance = InterpretedInstance<'static, ExecutionCtx>;

    fn execution_mode() -> &'static str {
        #[cfg(feature = "aot")]
        return "AOT pure";
        #[cfg(feature = "rvr")]
        return "RVR pure";
        #[cfg(all(not(feature = "aot"), not(feature = "rvr")))]
        return "Interpreted pure";
    }

    fn cache() -> &'static Cache<Self::Instance> {
        static CACHE: Cache<<PureExecution as BenchExecutor>::Instance> = OnceLock::new();
        &CACHE
    }

    fn build_instance(exe: &VmExe<BabyBear>) -> Self::Instance {
        Self::unwrap_instance(executor().instance(exe))
    }

    fn run_execution(instance: &Self::Instance, input: Vec<Vec<u8>>) -> Result<(), ExecutionError> {
        instance.execute(input, None).map(|_| ())
    }
}

impl BenchExecutor for MeteredExecution {
    #[cfg(feature = "aot")]
    type Instance = (AotInstance<'static, MeteredCtx>, MeteredCtx);
    #[cfg(feature = "rvr")]
    type Instance = (RvrMeteredInstance<'static, BabyBear>, MeteredCtx);
    #[cfg(all(not(feature = "aot"), not(feature = "rvr")))]
    type Instance = (InterpretedInstance<'static, MeteredCtx>, MeteredCtx);

    fn execution_mode() -> &'static str {
        #[cfg(feature = "aot")]
        return "AOT metered";
        #[cfg(feature = "rvr")]
        return "RVR metered";
        #[cfg(all(not(feature = "aot"), not(feature = "rvr")))]
        return "Interpreted metered";
    }

    fn cache() -> &'static Cache<Self::Instance> {
        static CACHE: Cache<<MeteredExecution as BenchExecutor>::Instance> = OnceLock::new();
        &CACHE
    }

    fn build_instance(exe: &VmExe<BabyBear>) -> Self::Instance {
        let (ctx, executor_idx_to_air_idx) = build_metered_ctx_for(exe);
        let instance =
            Self::unwrap_instance(executor().metered_instance(exe, &executor_idx_to_air_idx));
        (instance, ctx)
    }

    fn run_execution(instance: &Self::Instance, input: Vec<Vec<u8>>) -> Result<(), ExecutionError> {
        let (instance, ctx) = instance;
        instance.execute_metered(input, ctx.clone()).map(|_| ())
    }
}

impl BenchExecutor for MeteredCostExecution {
    #[cfg(feature = "rvr")]
    type Instance = RvrMeteredCostInstance<'static, BabyBear>;
    #[cfg(not(feature = "rvr"))]
    type Instance = InterpretedInstance<'static, MeteredCostCtx>;

    fn execution_mode() -> &'static str {
        #[cfg(feature = "rvr")]
        return "RVR metered cost";
        #[cfg(not(feature = "rvr"))]
        return "Interpreted metered cost";
    }

    fn cache() -> &'static Cache<Self::Instance> {
        static CACHE: Cache<<MeteredCostExecution as BenchExecutor>::Instance> = OnceLock::new();
        &CACHE
    }

    fn build_instance(exe: &VmExe<BabyBear>) -> Self::Instance {
        let (_ctx, executor_idx_to_air_idx) = metered_cost_setup();
        #[cfg(feature = "rvr")]
        let result = executor().metered_cost_instance(exe, executor_idx_to_air_idx, &_ctx.widths);
        #[cfg(not(feature = "rvr"))]
        let result = executor().metered_cost_instance(exe, executor_idx_to_air_idx);
        Self::unwrap_instance(result)
    }

    fn run_execution(instance: &Self::Instance, input: Vec<Vec<u8>>) -> Result<(), ExecutionError> {
        instance
            .execute_metered_cost(input, metered_cost_setup().0.clone())
            .map(|_| ())
    }
}

#[divan::bench(args = APP_PROGRAMS, sample_count=10)]
fn benchmark_execute(bencher: Bencher, program: &str) {
    PureExecution::benchmark(bencher, program);
}

#[divan::bench(args = APP_PROGRAMS, sample_count=5)]
fn benchmark_execute_metered(bencher: Bencher, program: &str) {
    MeteredExecution::benchmark(bencher, program);
}

#[divan::bench(ignore = true, args = APP_PROGRAMS, sample_count=5)]
fn benchmark_execute_metered_cost(bencher: Bencher, program: &str) {
    MeteredCostExecution::benchmark(bencher, program);
}
