use std::{any::Any, collections::BTreeMap};

use openvm_algebra_circuit::AlgebraPreflightGpuTracegen;
use openvm_bigint_circuit::Int256PreflightGpuTracegen;
use openvm_circuit::arch::{
    cuda::postflight::{
        GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
    },
    instructions::program::Program,
    prepare_gpu_postflight, GenerationError, Postflight, PostflightTracegen, PreflightOutput,
    VirtualMachine,
};
#[cfg(feature = "rvr")]
use openvm_circuit::arch::{
    rvr::cuda::{CheckpointReplayProgram, PostflightAccessRegistry},
    PreflightExecution,
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend};
use openvm_deferral_circuit::DeferralPreflightGpuTracegen;
use openvm_ecc_circuit::WeierstrassPreflightGpuTracegen;
use openvm_keccak256_circuit::Keccak256PreflightGpuTracegen;
use openvm_riscv_circuit::Rv64ImPreflightGpuTracegen;
use openvm_sha2_circuit::Sha2PreflightGpuTracegen;
use openvm_stark_backend::{
    p3_field::PrimeField32,
    prover::{AirProvingContext, ProvingContext},
    StarkEngine,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::{SdkVmConfig, SdkVmGpuBuilder};

/// One standard-SDK preflight GPU tracegen pass.
///
/// Extension producers remain concrete and finite. This coordinator only
/// installs their access schedules once, checks disjoint opcode ownership, and
/// visits the existing VM inventory once in reverse order. Pairing hints are
/// PHANTOM instructions, so their program row remains owned by the system
/// Phantom chip rather than by a pairing-specific producer.
struct SdkPreflightGpuTracegen<'a> {
    program: &'a GpuPostflightProgram,
    transcript: &'a GpuPostflightTranscript,
    replay_plan: &'a GpuPostflightPlan,
    rv64: Rv64ImPreflightGpuTracegen<'a>,
    keccak: Option<Keccak256PreflightGpuTracegen<'a>>,
    sha2: Option<Sha2PreflightGpuTracegen<'a>>,
    bigint: Option<Int256PreflightGpuTracegen<'a>>,
    algebra: Option<AlgebraPreflightGpuTracegen<'a>>,
    ecc: Option<WeierstrassPreflightGpuTracegen<'a>>,
    deferral: Option<DeferralPreflightGpuTracegen<'a>>,
}

impl SdkVmGpuBuilder {
    /// Uploads the immutable program used by interpreter preflight postflight
    /// and trace generation.
    #[cfg(not(feature = "rvr"))]
    pub(crate) fn upload_preflight_program<F: PrimeField32>(
        vm: &VirtualMachine<BabyBearPoseidon2GpuEngine, Self>,
        program: &Program<F>,
    ) -> Result<GpuPostflightProgram, GpuPostflightError> {
        let config = vm.config().to_inner();
        validate_preflight_config(
            config.modular.is_some(),
            config.fp2.is_some(),
            config.ecc.is_some(),
        )?;
        GpuPostflightProgram::upload(
            program,
            &vm.config().as_ref().memory_config,
            &vm.engine.device().device_ctx,
        )
    }

    /// Uploads one immutable program together with all postflight access
    /// schedules enabled by this SDK configuration.
    #[cfg(feature = "rvr")]
    pub(crate) fn upload_preflight_program<F: PrimeField32>(
        vm: &VirtualMachine<BabyBearPoseidon2GpuEngine, Self>,
        program: &Program<F>,
    ) -> Result<CheckpointReplayProgram, GpuPostflightError> {
        let config = vm.config().to_inner();
        validate_preflight_config(
            config.modular.is_some(),
            config.fp2.is_some(),
            config.ecc.is_some(),
        )?;
        let mut registry = PostflightAccessRegistry::default();
        if config.keccak.is_some() {
            Keccak256PreflightGpuTracegen::register_postflight_access_schedules(&mut registry)?;
        }
        if config.sha2.is_some() {
            Sha2PreflightGpuTracegen::register_postflight_access_schedules(&mut registry)?;
        }
        if config.bigint.is_some() {
            Int256PreflightGpuTracegen::register_postflight_access_schedules(&mut registry)?;
        }
        if let Some(modular) = &config.modular {
            AlgebraPreflightGpuTracegen::validate_postflight_program(program, modular)?;
            AlgebraPreflightGpuTracegen::register_postflight_access_schedules(
                &mut registry,
                modular,
                config.fp2.as_ref(),
            )?;
        }
        if let Some(ecc) = &config.ecc {
            WeierstrassPreflightGpuTracegen::register_postflight_access_schedules(
                &mut registry,
                ecc,
            )?;
        }
        if config.deferral.is_some() {
            DeferralPreflightGpuTracegen::register_postflight_access_schedules(&mut registry)?;
        }
        let native = Rv64ImPreflightGpuTracegen::postflight_opcode_bases();
        registry.validate_no_native_collisions(native)?;
        CheckpointReplayProgram::upload_with_postflight_access_registry(
            program,
            &vm.config().as_ref().memory_config,
            &registry,
            &vm.engine.device().device_ctx,
        )
    }

    /// Expands one metered preflight segment using the standard RV64/system
    /// chronology. The executor must already have enforced the segment's
    /// retired-instruction boundary.
    ///
    /// This layer deliberately does not guess executor buffer limits. The
    /// segment's metered instruction and residual counts must be used when
    /// constructing `PreflightLimits`.
    #[cfg(feature = "rvr")]
    pub(crate) fn postflight(
        vm: &VirtualMachine<BabyBearPoseidon2GpuEngine, Self>,
        program: &CheckpointReplayProgram,
        execution: &PreflightExecution,
        num_insns: u32,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError> {
        let result = vm.postflight(
            program,
            execution,
            num_insns,
            Rv64ImPreflightGpuTracegen::postflight_opcode_bases(),
        );
        #[cfg(feature = "metrics")]
        if let Ok((_, replay_plan)) = &result {
            vm.emit_preflight_opcode_counts(replay_plan);
        }
        result
    }

    /// Generates the standard SDK proving context from one postflight segment
    /// directly from the immutable preflight history.
    pub(crate) fn generate_preflight_proving_ctx(
        vm: &mut VirtualMachine<BabyBearPoseidon2GpuEngine, Self>,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        let max_trace_height = 1usize << vm.engine.params().log_stacked_height();
        SdkPreflightGpuTracegen::new(
            vm.config(),
            program,
            transcript,
            replay_plan,
            max_trace_height,
        )
        .map_err(extension_error)?
        .generate_proving_ctx(vm)
    }
}

impl PostflightTracegen<BabyBearPoseidon2GpuEngine> for SdkVmGpuBuilder {
    type Prepared = GpuPostflightProgram;

    fn prepare_postflight(
        vm: &VirtualMachine<BabyBearPoseidon2GpuEngine, Self>,
        program: &Program<BabyBear>,
    ) -> Result<Self::Prepared, GenerationError> {
        prepare_gpu_postflight(vm, program)
    }

    fn generate_proving_ctx(
        vm: &mut VirtualMachine<BabyBearPoseidon2GpuEngine, Self>,
        program: &Self::Prepared,
        output: &PreflightOutput,
        _postflight: &Postflight<'_, BabyBear>,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        let (transcript, replay_plan) = vm
            .postflight_history(program, output)
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        Self::generate_preflight_proving_ctx(vm, program, &transcript, &replay_plan)
    }
}

impl<'a> SdkPreflightGpuTracegen<'a> {
    fn new(
        config: &SdkVmConfig,
        program: &'a GpuPostflightProgram,
        transcript: &'a GpuPostflightTranscript,
        replay_plan: &'a GpuPostflightPlan,
        max_trace_height: usize,
    ) -> Result<Self, GpuPostflightError> {
        validate_preflight_config(
            config.modular.is_some(),
            config.fp2.is_some(),
            config.ecc.is_some(),
        )?;
        let keccak = config
            .keccak
            .as_ref()
            .map(|_| Keccak256PreflightGpuTracegen::new(program, transcript, replay_plan));
        let sha2 = config
            .sha2
            .as_ref()
            .map(|_| Sha2PreflightGpuTracegen::new(program, transcript, replay_plan));
        let bigint = config
            .bigint
            .as_ref()
            .map(|_| Int256PreflightGpuTracegen::new(program, transcript, replay_plan));
        let algebra = config
            .modular
            .as_ref()
            .map(|modular| {
                AlgebraPreflightGpuTracegen::new(
                    program,
                    transcript,
                    replay_plan,
                    modular,
                    config.fp2.as_ref(),
                )
            })
            .transpose()?;
        let ecc = config
            .ecc
            .as_ref()
            .map(|ecc| WeierstrassPreflightGpuTracegen::new(ecc, program, transcript, replay_plan));
        let deferral = config
            .deferral
            .as_ref()
            .map(|_| {
                DeferralPreflightGpuTracegen::new(
                    program,
                    transcript,
                    replay_plan,
                    max_trace_height,
                )
            })
            .transpose()?;

        let mut ownership = OpcodeOwnership::new();
        if keccak.is_some() {
            ownership.claim("Keccak", Keccak256PreflightGpuTracegen::extension_opcodes())?;
        }
        if sha2.is_some() {
            ownership.claim("SHA-2", Sha2PreflightGpuTracegen::extension_opcodes())?;
        }
        if bigint.is_some() {
            ownership.claim("Int256", Int256PreflightGpuTracegen::extension_opcodes())?;
        }
        if let Some(algebra) = &algebra {
            ownership.claim("Algebra", algebra.extension_opcodes().iter().copied())?;
        }
        if let Some(ecc) = &ecc {
            ownership.claim("Weierstrass", ecc.claimed_opcodes().iter().copied())?;
        }
        if deferral.is_some() {
            ownership.claim(
                "Deferral",
                DeferralPreflightGpuTracegen::extension_opcodes(),
            )?;
        }
        ownership.validate_executed(replay_plan.executed_opcodes())?;
        let extension_opcodes = ownership.extension_opcodes();
        let rv64 = Rv64ImPreflightGpuTracegen::new_after_claiming_extension_opcodes(
            program,
            transcript,
            replay_plan,
            &extension_opcodes,
        )?;

        Ok(Self {
            program,
            transcript,
            replay_plan,
            rv64,
            keccak,
            sha2,
            bigint,
            algebra,
            ecc,
            deferral,
        })
    }

    /// Generates every standard SDK AIR through one reverse inventory walk.
    fn generate_proving_ctx(
        self,
        vm: &mut VirtualMachine<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        vm.generate_preflight_proving_ctx(
            self.program,
            self.transcript,
            self.replay_plan,
            self,
            |tracegen, insertion_idx, chip| tracegen.generate_for_chip(insertion_idx, chip),
            SdkPreflightGpuTracegen::finish,
        )
    }

    fn generate_for_chip(
        &mut self,
        insertion_idx: usize,
        chip: &dyn Any,
    ) -> Result<AirProvingContext<GpuBackend>, GenerationError> {
        if let Some(tracegen) = &mut self.deferral {
            if let Some(ctx) = tracegen.generate_for_chip(chip).map_err(extension_error)? {
                return Ok(ctx);
            }
        }
        if let Some(tracegen) = &mut self.keccak {
            if let Some(ctx) = tracegen.generate_for_chip(chip).map_err(extension_error)? {
                return Ok(ctx);
            }
        }
        if let Some(tracegen) = &mut self.sha2 {
            if let Some(ctx) = tracegen.generate_for_chip(chip).map_err(extension_error)? {
                return Ok(ctx);
            }
        }
        if let Some(tracegen) = &mut self.bigint {
            if let Some(ctx) = tracegen.generate_for_chip(chip).map_err(extension_error)? {
                return Ok(ctx);
            }
        }
        if let Some(tracegen) = &mut self.ecc {
            if let Some(ctx) = tracegen.generate_for_chip(chip).map_err(extension_error)? {
                return Ok(ctx);
            }
        }
        if let Some(tracegen) = &mut self.algebra {
            if let Some(ctx) = tracegen.generate_for_chip(chip).map_err(extension_error)? {
                return Ok(ctx);
            }
        }
        self.rv64
            .generate_for_chip(insertion_idx, chip)
            .map_err(extension_error)
    }

    fn finish(self) -> Result<(), GenerationError> {
        if let Some(tracegen) = self.keccak {
            tracegen.finish().map_err(extension_error)?;
        }
        if let Some(tracegen) = self.sha2 {
            tracegen.finish().map_err(extension_error)?;
        }
        if let Some(tracegen) = self.bigint {
            tracegen.finish().map_err(extension_error)?;
        }
        if let Some(tracegen) = self.ecc {
            tracegen.finish().map_err(extension_error)?;
        }
        if let Some(tracegen) = self.deferral {
            tracegen.finish().map_err(extension_error)?;
        }
        if let Some(tracegen) = self.algebra {
            tracegen.finish().map_err(extension_error)?;
        }
        self.rv64.finish().map_err(extension_error)
    }
}

fn extension_error(error: GpuPostflightError) -> GenerationError {
    GenerationError::ExtensionTracegen(error.to_string())
}

fn validate_preflight_config(
    has_modular: bool,
    has_fp2: bool,
    has_ecc: bool,
) -> Result<(), GpuPostflightError> {
    if !has_modular && has_fp2 {
        return Err(GpuPostflightError::InvalidAccessSchedule(
            "Fp2 preflight replay requires the Modular extension".to_string(),
        ));
    }
    if !has_modular && has_ecc {
        return Err(GpuPostflightError::InvalidAccessSchedule(
            "Weierstrass preflight replay requires the Modular extension".to_string(),
        ));
    }
    Ok(())
}

struct OpcodeOwnership {
    extensions: BTreeMap<u32, &'static str>,
}

impl OpcodeOwnership {
    fn new() -> Self {
        Self {
            extensions: BTreeMap::new(),
        }
    }

    fn claim(
        &mut self,
        owner: &'static str,
        opcodes: impl IntoIterator<Item = u32>,
    ) -> Result<(), GpuPostflightError> {
        for opcode in opcodes {
            if Rv64ImPreflightGpuTracegen::owns_opcode(opcode) {
                return Err(GpuPostflightError::InvalidTranscript(format!(
                    "{owner} opcode {opcode:#x} collides with RV64/system"
                )));
            }
            if let Some(previous) = self.extensions.get(&opcode) {
                return Err(GpuPostflightError::InvalidTranscript(format!(
                    "opcode {opcode:#x} is owned by both {previous} and {owner}"
                )));
            }
            self.extensions.insert(opcode, owner);
        }
        Ok(())
    }

    fn validate_executed(
        &self,
        executed: impl IntoIterator<Item = u32>,
    ) -> Result<(), GpuPostflightError> {
        if let Some(opcode) = executed.into_iter().find(|opcode| {
            !Rv64ImPreflightGpuTracegen::owns_opcode(*opcode)
                && !self.extensions.contains_key(opcode)
        }) {
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "executed opcode {opcode:#x} has no standard SDK preflight trace producer"
            )));
        }
        Ok(())
    }

    fn extension_opcodes(&self) -> Vec<u32> {
        self.extensions.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "rvr")]
    use openvm_circuit::arch::{
        MemoryConfig, PreflightEndpoint, PreflightLimits, SystemConfig, VmExecutor,
    };
    #[cfg(feature = "rvr")]
    use openvm_instructions::{
        exe::VmExe,
        instruction::Instruction,
        program::Program,
        riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
        PUBLIC_VALUES_AS,
    };
    use openvm_instructions::{LocalOpcode, SystemOpcode};
    #[cfg(feature = "rvr")]
    use openvm_stark_backend::SystemParams;
    #[cfg(feature = "rvr")]
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::*;

    #[cfg(feature = "rvr")]
    fn small_system_config() -> SystemConfig {
        let mut address_spaces = MemoryConfig::empty_address_space_configs(5);
        address_spaces[RV64_REGISTER_AS as usize].num_cells = 1 << 12;
        address_spaces[RV64_MEMORY_AS as usize].num_cells = 1 << 22;
        address_spaces[PUBLIC_VALUES_AS as usize].num_cells = 1 << 12;
        SystemConfig::new(3, MemoryConfig::new(2, address_spaces, 29, 29, 17), 32)
    }

    #[cfg(feature = "rvr")]
    #[test]
    fn standard_sdk_inventory_proves_from_record_free_preflight() {
        let program = Program::from_instructions(&[Instruction::<BabyBear>::from_usize(
            SystemOpcode::TERMINATE.global_opcode(),
            [0; 7],
        )]);
        let exe = VmExe::new(program.clone());
        let mut config = SdkVmConfig::standard();
        config.system.config = small_system_config();
        let executor = VmExecutor::new(config.clone()).unwrap();
        let preflight = executor.preflight_instance(&exe).unwrap();
        let state = preflight.create_initial_vm_state(Vec::<Vec<u8>>::new());

        let mut params = SystemParams::new_for_testing(21);
        params.max_constraint_degree = 3;
        let (mut vm, pk) = VirtualMachine::new_with_keygen(
            BabyBearPoseidon2GpuEngine::new(params),
            SdkVmGpuBuilder,
            config,
        )
        .unwrap();
        let cached_program = vm.commit_program_on_device(&program);
        vm.load_program(cached_program);
        vm.transport_init_memory_to_device(&state.memory);
        let gpu_program = SdkVmGpuBuilder::upload_preflight_program(&vm, &program).unwrap();
        let execution = preflight
            .execute_from_state(state, PreflightLimits::new(1, 0, 1))
            .unwrap();
        assert_eq!(execution.endpoint, PreflightEndpoint::Terminated);
        let (transcript, replay_plan) =
            SdkVmGpuBuilder::postflight(&vm, &gpu_program, &execution, execution.retired).unwrap();
        let proving_ctx = SdkVmGpuBuilder::generate_preflight_proving_ctx(
            &mut vm,
            gpu_program.program(),
            &transcript,
            &replay_plan,
        )
        .unwrap();
        drop(replay_plan);
        drop(transcript);
        let proof = vm.engine.prove(vm.pk(), proving_ctx).unwrap();
        vm.engine.verify(&pk.get_vk(), &proof).unwrap();
    }

    #[test]
    fn opcode_ownership_rejects_duplicates_and_missing_producers() {
        let mut free_opcodes = (0..=u16::MAX as u32)
            .filter(|opcode| !Rv64ImPreflightGpuTracegen::owns_opcode(*opcode));
        let claimed_opcode = free_opcodes.next().unwrap();
        let missing_opcode = free_opcodes.next().unwrap();
        let mut ownership = OpcodeOwnership::new();
        ownership.claim("first", [claimed_opcode]).unwrap();

        let duplicate = ownership.claim("second", [claimed_opcode]).unwrap_err();
        assert!(duplicate.to_string().contains("owned by both"));

        let native_collision = ownership
            .claim(
                "extension",
                [SystemOpcode::PHANTOM.global_opcode().as_usize() as u32],
            )
            .unwrap_err();
        assert!(native_collision.to_string().contains("RV64/system"));

        let missing = ownership.validate_executed([missing_opcode]).unwrap_err();
        assert!(missing
            .to_string()
            .contains("has no standard SDK preflight trace producer"));

        ownership
            .validate_executed([
                SystemOpcode::TERMINATE.global_opcode().as_usize() as u32,
                claimed_opcode,
            ])
            .unwrap();
    }
}
