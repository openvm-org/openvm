use std::{
    fs, io,
    path::Path,
    sync::{Arc, OnceLock},
};

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
        execution_mode::{MeteredCostCtx, MeteredCtx},
        instructions::exe::VmExe,
        interpreter::InterpretedInstance,
        InitFileGenerator, SystemConfig, VirtualMachine, VmExecutor,
    },
    derive::VmConfig,
    system::*,
};
use openvm_continuations::{
    verifier::{internal::types::InternalVmVerifierInput, leaf::types::LeafVmVerifierInput},
    SC,
};
use openvm_ecc_circuit::{EccCpuProverExt, WeierstrassExtension, WeierstrassExtensionExecutor};
use openvm_ecc_transpiler::EccTranspilerExtension;
use openvm_keccak256_circuit::{Keccak256, Keccak256CpuProverExt, Keccak256Executor};
use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_native_circuit::NativeCpuBuilder;
use openvm_native_recursion::hints::Hintable;
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
use openvm_sdk::{
    commit::VmCommittedExe,
    config::{AggregationConfig, DEFAULT_NUM_CHILDREN_INTERNAL, DEFAULT_NUM_CHILDREN_LEAF},
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
        proof::Proof,
        prover::{
            cpu::{CpuBackend, CpuDevice},
            hal::DeviceDataTransporter,
        },
    },
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::{transpiler::Transpiler, FromElf};
use serde::{Deserialize, Serialize};

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
            system: SystemConfig::default(),
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
    ) -> io::Result<()> {
        Ok(())
    }
}

pub struct ExecuteBuilder;
impl<E, SC> openvm_circuit::arch::VmBuilder<E> for ExecuteBuilder
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
{
    type VmConfig = ExecuteConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = openvm_circuit::arch::MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &ExecuteConfig,
        circuit: openvm_circuit::arch::AirInventory<SC>,
    ) -> Result<
        openvm_circuit::arch::VmChipComplex<
            SC,
            Self::RecordArena,
            E::PB,
            Self::SystemChipInventory,
        >,
        openvm_circuit::arch::ChipInventoryError,
    > {
        use openvm_circuit::arch::VmBuilder;
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemCpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        openvm_circuit::arch::VmProverExtension::<E, _, _>::extend_prover(
            &Rv32ImCpuProverExt,
            &config.rv32i,
            inventory,
        )?;
        openvm_circuit::arch::VmProverExtension::<E, _, _>::extend_prover(
            &Rv32ImCpuProverExt,
            &config.rv32m,
            inventory,
        )?;
        openvm_circuit::arch::VmProverExtension::<E, _, _>::extend_prover(
            &Rv32ImCpuProverExt,
            &config.io,
            inventory,
        )?;
        openvm_circuit::arch::VmProverExtension::<E, _, _>::extend_prover(
            &Int256CpuProverExt,
            &config.bigint,
            inventory,
        )?;
        openvm_circuit::arch::VmProverExtension::<E, _, _>::extend_prover(
            &Keccak256CpuProverExt,
            &config.keccak,
            inventory,
        )?;
        openvm_circuit::arch::VmProverExtension::<E, _, _>::extend_prover(
            &Sha2CpuProverExt,
            &config.sha256,
            inventory,
        )?;
        openvm_circuit::arch::VmProverExtension::<E, _, _>::extend_prover(
            &AlgebraCpuProverExt,
            &config.modular,
            inventory,
        )?;
        openvm_circuit::arch::VmProverExtension::<E, _, _>::extend_prover(
            &AlgebraCpuProverExt,
            &config.fp2,
            inventory,
        )?;
        openvm_circuit::arch::VmProverExtension::<E, _, _>::extend_prover(
            &EccCpuProverExt,
            &config.weierstrass,
            inventory,
        )?;
        openvm_circuit::arch::VmProverExtension::<E, _, _>::extend_prover(
            &PairingProverExt,
            &config.pairing,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

pub type NativeVm = VirtualMachine<BabyBearPoseidon2Engine, NativeCpuBuilder>;

static METERED_CTX: OnceLock<(MeteredCtx, Vec<usize>)> = OnceLock::new();
static METERED_COST_CTX: OnceLock<(MeteredCostCtx, Vec<usize>)> = OnceLock::new();
static EXECUTOR: OnceLock<VmExecutor<BabyBear, ExecuteConfig>> = OnceLock::new();

pub fn create_default_transpiler() -> Transpiler<BabyBear> {
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

pub fn load_program_executable(program: &str) -> eyre::Result<VmExe<BabyBear>> {
    let transpiler = create_default_transpiler();
    let program_dir = get_programs_dir().join(program);
    let elf_path = get_elf_path(&program_dir);
    let elf = read_elf_file(&elf_path)?;
    Ok(VmExe::from_elf(elf, transpiler)?)
}

pub fn metering_setup() -> &'static (MeteredCtx, Vec<usize>) {
    METERED_CTX.get_or_init(|| {
        let config = ExecuteConfig::default();
        let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
        let (vm, _) = VirtualMachine::new_with_keygen(engine, ExecuteBuilder, config).unwrap();
        let ctx = vm.build_metered_ctx();
        let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
        (ctx, executor_idx_to_air_idx)
    })
}

pub fn metered_cost_setup() -> &'static (MeteredCostCtx, Vec<usize>) {
    METERED_COST_CTX.get_or_init(|| {
        let config = ExecuteConfig::default();
        let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
        let (vm, _) = VirtualMachine::new_with_keygen(engine, ExecuteBuilder, config).unwrap();
        let ctx = vm.build_metered_cost_ctx();
        let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
        (ctx, executor_idx_to_air_idx)
    })
}

pub fn executor() -> &'static VmExecutor<BabyBear, ExecuteConfig> {
    EXECUTOR.get_or_init(|| {
        let vm_config = ExecuteConfig::default();
        VmExecutor::<BabyBear, _>::new(vm_config).unwrap()
    })
}

pub fn setup_leaf_verifier(program: &str) -> (NativeVm, VmExe<BabyBear>, Vec<Vec<BabyBear>>) {
    use openvm_benchmarks_utils::get_fixtures_dir;

    let fixtures_dir = get_fixtures_dir();

    let app_proof_bytes = fs::read(fixtures_dir.join(format!("{}.app.proof", program))).unwrap();
    let app_proof: openvm_circuit::arch::ContinuationVmProof<SC> =
        bitcode::deserialize(&app_proof_bytes).unwrap();

    let leaf_exe_bytes = fs::read(fixtures_dir.join(format!("{}.leaf.exe", program))).unwrap();
    let leaf_exe: VmExe<BabyBear> = bitcode::deserialize(&leaf_exe_bytes).unwrap();

    let leaf_pk_bytes = fs::read(fixtures_dir.join(format!("{}.leaf.pk", program))).unwrap();
    let leaf_pk = bitcode::deserialize(&leaf_pk_bytes).unwrap();

    let leaf_inputs =
        LeafVmVerifierInput::chunk_continuation_vm_proof(&app_proof, DEFAULT_NUM_CHILDREN_LEAF);
    let leaf_input = leaf_inputs.first().expect("No leaf input available");

    let agg_config = AggregationConfig::default();
    let config = agg_config.leaf_vm_config();
    let engine = BabyBearPoseidon2Engine::new(agg_config.leaf_fri_params);
    use openvm_stark_sdk::engine::StarkEngine;
    let d_pk = engine.device().transport_pk_to_device(&leaf_pk);
    let vm = VirtualMachine::new(engine, NativeCpuBuilder, config, d_pk).unwrap();
    let input_stream = leaf_input.write_to_stream();

    (vm, leaf_exe, input_stream)
}

pub fn setup_internal_verifier(
    program: &str,
) -> (NativeVm, Arc<VmExe<BabyBear>>, Vec<Vec<BabyBear>>) {
    use openvm_benchmarks_utils::get_fixtures_dir;

    let fixtures_dir = get_fixtures_dir();

    let internal_exe_bytes =
        fs::read(fixtures_dir.join(format!("{}.internal.exe", program))).unwrap();
    let internal_exe: VmExe<BabyBear> = bitcode::deserialize(&internal_exe_bytes).unwrap();

    let internal_pk_bytes =
        fs::read(fixtures_dir.join(format!("{}.internal.pk", program))).unwrap();
    let internal_pk = bitcode::deserialize(&internal_pk_bytes).unwrap();

    // Load leaf proof by index (using index 0)
    let leaf_proof_bytes = fs::read(fixtures_dir.join(format!("{}.leaf.0.proof", program)))
        .expect("No leaf proof available at index 0");
    let leaf_proof: Proof<SC> = bitcode::deserialize(&leaf_proof_bytes).unwrap();

    let agg_config = AggregationConfig::default();
    let config = agg_config.internal_vm_config();
    let engine = BabyBearPoseidon2Engine::new(agg_config.internal_fri_params);

    let internal_committed_exe = VmCommittedExe::<SC>::commit(internal_exe, engine.config().pcs());
    let internal_inputs = InternalVmVerifierInput::chunk_leaf_or_internal_proofs(
        internal_committed_exe.get_program_commit().into(),
        &[leaf_proof],
        DEFAULT_NUM_CHILDREN_INTERNAL,
    );

    use openvm_stark_sdk::engine::StarkEngine;
    let d_pk = engine.device().transport_pk_to_device(&internal_pk);
    let vm = VirtualMachine::new(engine, NativeCpuBuilder, config, d_pk).unwrap();
    let input_stream = internal_inputs.first().unwrap().write();

    (vm, internal_committed_exe.exe, input_stream)
}

pub fn transmute_interpreter_lifetime<'a, Ctx>(
    interpreter: InterpretedInstance<'_, BabyBear, Ctx>,
) -> InterpretedInstance<'a, BabyBear, Ctx> {
    // SAFETY: We transmute the interpreter to have the same lifetime as the VM.
    // This is safe because the vm is moved into the tuple and will remain
    // alive for the entire duration that the interpreter is used.
    unsafe { std::mem::transmute(interpreter) }
}
