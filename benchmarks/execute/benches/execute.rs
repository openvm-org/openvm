use std::{fs, path::Path, sync::OnceLock};

use divan::Bencher;
use eyre::Result;
use openvm_algebra_circuit::{
    AlgebraCpuProverExt, Fp2Extension, Fp2ExtensionExecutor, ModularExtension,
    ModularExtensionExecutor,
};
use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use openvm_benchmarks_utils::{get_elf_path, get_fixtures_dir, get_programs_dir, read_elf_file};
use openvm_bigint_circuit::{Int256, Int256CpuProverExt, Int256Executor};
use openvm_bigint_transpiler::Int256TranspilerExtension;
use openvm_circuit::{
    arch::{
        execution_mode::{MeteredCostCtx, MeteredCtx},
        instructions::exe::VmExe,
        interpreter::InterpretedInstance,
        ContinuationVmProof, *,
    },
    derive::VmConfig,
    system::*,
};
use openvm_continuations::{
    verifier::{common::types::VmVerifierPvs, leaf::types::LeafVmVerifierInput},
    SC,
};
use openvm_ecc_circuit::{EccCpuProverExt, WeierstrassExtension, WeierstrassExtensionExecutor};
use openvm_ecc_transpiler::EccTranspilerExtension;
use openvm_keccak256_circuit::{Keccak256, Keccak256CpuProverExt, Keccak256Executor};
use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_native_circuit::{NativeConfig, NativeCpuBuilder, NATIVE_MAX_TRACE_HEIGHTS};
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
use openvm_sdk::config::{DEFAULT_LEAF_LOG_BLOWUP, SBOX_SIZE};
use openvm_sha256_circuit::{Sha256, Sha256Executor, Sha2CpuProverExt};
use openvm_sha256_transpiler::Sha256TranspilerExtension;
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    engine::{StarkEngine, StarkFriEngine},
    openvm_stark_backend::{
        self,
        config::{StarkGenericConfig, Val},
        p3_field::PrimeField32,
        prover::{
            cpu::{CpuBackend, CpuDevice},
            hal::DeviceDataTransporter,
        },
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
static METERED_COST_CTX: OnceLock<(MeteredCostCtx, Vec<usize>)> = OnceLock::new();
static EXECUTOR: OnceLock<VmExecutor<BabyBear, ExecuteConfig>> = OnceLock::new();

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

fn metered_cost_setup() -> &'static (MeteredCostCtx, Vec<usize>) {
    METERED_COST_CTX.get_or_init(|| {
        let config = ExecuteConfig::default();
        let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
        let (vm, _) = VirtualMachine::new_with_keygen(engine, ExecuteBuilder, config).unwrap();
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

#[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=10)]
fn benchmark_execute(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let exe = load_program_executable(program).expect("Failed to load program executable");
            let interpreter = executor().instance(&exe).unwrap();
            (interpreter, vec![])
        })
        .bench_values(|(interpreter, input)| {
            interpreter
                .execute(input, None)
                .expect("Failed to execute program in interpreted mode");
        });
}

#[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=5)]
fn benchmark_execute_metered(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let exe = load_program_executable(program).expect("Failed to load program executable");
            let (ctx, executor_idx_to_air_idx) = metering_setup();
            let interpreter = executor()
                .metered_instance(&exe, executor_idx_to_air_idx)
                .unwrap();
            (interpreter, vec![], ctx.clone())
        })
        .bench_values(|(interpreter, input, ctx)| {
            interpreter
                .execute_metered(input, ctx)
                .expect("Failed to execute program");
        });
}

#[divan::bench(ignore = true, args = AVAILABLE_PROGRAMS, sample_count=5)]
fn benchmark_execute_metered_cost(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let exe = load_program_executable(program).expect("Failed to load program executable");
            let (ctx, executor_idx_to_air_idx) = metered_cost_setup();
            let interpreter = executor()
                .metered_cost_instance(&exe, executor_idx_to_air_idx)
                .unwrap();
            (interpreter, vec![], ctx.clone())
        })
        .bench_values(|(interpreter, input, ctx)| {
            interpreter
                .execute_metered_cost(input, ctx)
                .expect("Failed to execute program with metered cost");
        });
}

fn setup_leaf_verifier() -> (
    VirtualMachine<BabyBearPoseidon2Engine, NativeCpuBuilder>,
    VmExe<BabyBear>,
    Vec<Vec<BabyBear>>,
) {
    let fixtures_dir = get_fixtures_dir();
    let app_proof_bytes = fs::read(fixtures_dir.join("kitchen-sink.app.proof")).unwrap();
    let app_proof: ContinuationVmProof<SC> = bitcode::deserialize(&app_proof_bytes).unwrap();

    let leaf_exe_bytes = fs::read(fixtures_dir.join("kitchen-sink.leaf.exe")).unwrap();
    let leaf_exe: VmExe<BabyBear> = bitcode::deserialize(&leaf_exe_bytes).unwrap();

    let leaf_pk_bytes = fs::read(fixtures_dir.join("kitchen-sink.leaf.pk")).unwrap();
    let leaf_pk = bitcode::deserialize(&leaf_pk_bytes).unwrap();

    let leaf_inputs = LeafVmVerifierInput::chunk_continuation_vm_proof(&app_proof, 2);
    let leaf_input = leaf_inputs.first().expect("No leaf input available");

    let config = NativeConfig::aggregation(
        VmVerifierPvs::<u8>::width(),
        SBOX_SIZE.min(FriParameters::standard_fast().max_constraint_degree()),
    );
    let fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(DEFAULT_LEAF_LOG_BLOWUP);
    let engine = BabyBearPoseidon2Engine::new(fri_params);
    let d_pk = engine.device().transport_pk_to_device(&leaf_pk);
    let vm = VirtualMachine::new(engine, NativeCpuBuilder, config, d_pk).unwrap();
    let input_stream = leaf_input.write_to_stream();

    (vm, leaf_exe, input_stream)
}

#[divan::bench(sample_count = 5)]
fn benchmark_leaf_verifier_execute(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let (vm, leaf_exe, input_stream) = setup_leaf_verifier();
            let interpreter = vm.executor().instance(&leaf_exe).unwrap();

            // SAFETY: We transmute the interpreter to have the same lifetime as the VM.
            // This is safe because the vm is moved into the tuple and will remain
            // alive for the entire duration that the interpreter is used.
            #[allow(clippy::missing_transmute_annotations)]
            let interpreter =
                unsafe { std::mem::transmute::<_, InterpretedInstance<'_, _, _>>(interpreter) };

            (vm, interpreter, input_stream)
        })
        .bench_values(|(_vm, interpreter, input_stream)| {
            interpreter
                .execute(input_stream, None)
                .expect("Failed to execute program in interpreted mode");
        });
}

#[divan::bench(sample_count = 5)]
fn benchmark_leaf_verifier_execute_metered(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let (vm, leaf_exe, input_stream) = setup_leaf_verifier();
            let ctx = vm.build_metered_ctx();
            let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
            let interpreter = vm
                .executor()
                .metered_instance(&leaf_exe, &executor_idx_to_air_idx)
                .unwrap();

            // SAFETY: We transmute the interpreter to have the same lifetime as the VM.
            // This is safe because the vm is moved into the tuple and will remain
            // alive for the entire duration that the interpreter is used.
            #[allow(clippy::missing_transmute_annotations)]
            let interpreter =
                unsafe { std::mem::transmute::<_, InterpretedInstance<'_, _, _>>(interpreter) };

            (vm, interpreter, input_stream, ctx)
        })
        .bench_values(|(_vm, interpreter, input_stream, ctx)| {
            interpreter
                .execute_metered(input_stream, ctx)
                .expect("Failed to execute program");
        });
}

#[divan::bench(sample_count = 5)]
fn benchmark_leaf_verifier_execute_preflight(bencher: Bencher) {
    bencher
        .with_inputs(|| {
            let (vm, leaf_exe, input_stream) = setup_leaf_verifier();
            let state = vm.create_initial_state(&leaf_exe, input_stream);
            let interpreter = vm.preflight_interpreter(&leaf_exe).unwrap();

            (vm, state, interpreter)
        })
        .bench_values(|(vm, state, mut interpreter)| {
            let _out = vm
                .execute_preflight(&mut interpreter, state, None, NATIVE_MAX_TRACE_HEIGHTS)
                .expect("Failed to execute preflight");
        });
}
