use std::collections::BTreeMap;

use openvm_algebra_circuit::AlgebraRvrGpuTracegen;
use openvm_bigint_circuit::Int256RvrGpuTracegen;
use openvm_circuit::arch::{
    instructions::program::Program,
    rvr::{
        cuda::{
            GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript,
            RvrCheckpointAccessRegistry, RvrCheckpointOpcodeBases,
        },
        RvrCheckpointPreflightExecution,
    },
    GenerationError, VirtualMachine,
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend};
use openvm_deferral_circuit::DeferralRvrGpuTracegen;
use openvm_ecc_circuit::WeierstrassRvrGpuTracegen;
use openvm_keccak256_circuit::Keccak256RvrGpuTracegen;
use openvm_riscv_circuit::Rv64ImRvrGpuTracegen;
use openvm_sha2_circuit::Sha2RvrGpuTracegen;
use openvm_stark_backend::{p3_field::PrimeField32, prover::ProvingContext, StarkEngine};

use crate::{SdkVmConfig, SdkVmGpuBuilder};

/// One standard-SDK checkpoint tracegen pass.
///
/// Extension producers remain concrete and finite. This coordinator only
/// installs their access schedules once, checks disjoint opcode ownership, and
/// visits the existing VM inventory once in reverse order. Pairing hints are
/// PHANTOM instructions, so their program row remains owned by the system
/// Phantom chip rather than by a pairing-specific producer.
struct SdkRvrGpuTracegen<'a> {
    program: &'a GpuRvrProgram,
    transcript: &'a GpuRvrTranscript,
    replay_plan: &'a GpuRvrReplayPlan,
    rv64: Rv64ImRvrGpuTracegen<'a>,
    keccak: Option<Keccak256RvrGpuTracegen<'a>>,
    sha2: Option<Sha2RvrGpuTracegen<'a>>,
    bigint: Option<Int256RvrGpuTracegen<'a>>,
    algebra: Option<AlgebraRvrGpuTracegen<'a>>,
    ecc: Option<WeierstrassRvrGpuTracegen<'a>>,
    deferral: Option<DeferralRvrGpuTracegen<'a>>,
}

impl SdkVmGpuBuilder {
    /// Uploads one immutable program together with all checkpoint access
    /// schedules enabled by this SDK configuration.
    pub fn upload_checkpoint_program<F: PrimeField32>(
        vm: &VirtualMachine<BabyBearPoseidon2GpuEngine, Self>,
        program: &Program<F>,
    ) -> Result<GpuRvrProgram, GpuRvrInputError> {
        let config = vm.config().to_inner();
        validate_checkpoint_config(
            config.modular.is_some(),
            config.fp2.is_some(),
            config.ecc.is_some(),
        )?;
        let mut registry = RvrCheckpointAccessRegistry::default();
        if config.keccak.is_some() {
            Keccak256RvrGpuTracegen::register_checkpoint_access_schedules(&mut registry)?;
        }
        if config.sha2.is_some() {
            Sha2RvrGpuTracegen::register_checkpoint_access_schedules(&mut registry)?;
        }
        if config.bigint.is_some() {
            Int256RvrGpuTracegen::register_checkpoint_access_schedules(&mut registry)?;
        }
        if let Some(modular) = &config.modular {
            AlgebraRvrGpuTracegen::validate_checkpoint_program(program, modular)?;
            AlgebraRvrGpuTracegen::register_checkpoint_access_schedules(
                &mut registry,
                modular,
                config.fp2.as_ref(),
            )?;
        }
        if let Some(ecc) = &config.ecc {
            WeierstrassRvrGpuTracegen::register_checkpoint_access_schedules(&mut registry, ecc)?;
        }
        if config.deferral.is_some() {
            DeferralRvrGpuTracegen::register_checkpoint_access_schedules(&mut registry)?;
        }
        let native = Rv64ImRvrGpuTracegen::checkpoint_opcode_bases();
        registry.validate_no_native_collisions(native)?;
        GpuRvrProgram::upload_with_checkpoint_access_registry(
            program,
            &config.system.memory_config,
            &registry,
            &vm.engine.device().device_ctx,
        )
    }

    /// Expands one exact checkpoint segment using the standard RV64/system
    /// chronology. The executor must already have enforced the segment's exact
    /// retired-instruction boundary.
    ///
    /// This layer deliberately does not guess executor buffer limits. The
    /// segment's metered instruction and residual counts must be used when
    /// constructing `RvrCheckpointPreflightLimits`.
    pub fn expand_checkpoint_replay(
        vm: &VirtualMachine<BabyBearPoseidon2GpuEngine, Self>,
        program: &GpuRvrProgram,
        execution: &RvrCheckpointPreflightExecution,
        expected_retired: u32,
    ) -> Result<(GpuRvrTranscript, GpuRvrReplayPlan), GpuRvrInputError> {
        vm.expand_rvr_checkpoint_replay(
            program,
            execution,
            expected_retired,
            Rv64ImRvrGpuTracegen::checkpoint_opcode_bases(),
        )
    }

    /// Generates the standard SDK proving context from one expanded checkpoint
    /// segment without constructing a `RecordArena`.
    pub fn generate_proving_ctx_from_rvr(
        vm: &mut VirtualMachine<BabyBearPoseidon2GpuEngine, Self>,
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        replay_plan: &GpuRvrReplayPlan,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        let max_trace_height = 1usize << vm.engine.params().log_stacked_height();
        SdkRvrGpuTracegen::new(
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

impl<'a> SdkRvrGpuTracegen<'a> {
    fn new(
        config: &SdkVmConfig,
        program: &'a GpuRvrProgram,
        transcript: &'a GpuRvrTranscript,
        replay_plan: &'a GpuRvrReplayPlan,
        max_trace_height: usize,
    ) -> Result<Self, GpuRvrInputError> {
        validate_checkpoint_config(
            config.modular.is_some(),
            config.fp2.is_some(),
            config.ecc.is_some(),
        )?;
        let keccak = config
            .keccak
            .as_ref()
            .map(|_| Keccak256RvrGpuTracegen::new(program, transcript, replay_plan));
        let sha2 = config
            .sha2
            .as_ref()
            .map(|_| Sha2RvrGpuTracegen::new(program, transcript, replay_plan));
        let bigint = config
            .bigint
            .as_ref()
            .map(|_| Int256RvrGpuTracegen::new(program, transcript, replay_plan));
        let algebra = config
            .modular
            .as_ref()
            .map(|modular| {
                AlgebraRvrGpuTracegen::new(
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
            .map(|ecc| WeierstrassRvrGpuTracegen::new(ecc, program, transcript, replay_plan));
        let deferral = config
            .deferral
            .as_ref()
            .map(|_| {
                DeferralRvrGpuTracegen::new(program, transcript, replay_plan, max_trace_height)
            })
            .transpose()?;

        let native = Rv64ImRvrGpuTracegen::checkpoint_opcode_bases();
        let mut ownership = OpcodeOwnership::new(native);
        if keccak.is_some() {
            ownership.claim("Keccak", Keccak256RvrGpuTracegen::extension_opcodes())?;
        }
        if sha2.is_some() {
            ownership.claim("SHA-2", Sha2RvrGpuTracegen::extension_opcodes())?;
        }
        if bigint.is_some() {
            ownership.claim("Int256", Int256RvrGpuTracegen::extension_opcodes())?;
        }
        if let Some(algebra) = &algebra {
            ownership.claim("Algebra", algebra.extension_opcodes().iter().copied())?;
        }
        if let Some(ecc) = &ecc {
            ownership.claim("Weierstrass", ecc.claimed_opcodes().iter().copied())?;
        }
        if deferral.is_some() {
            ownership.claim("Deferral", DeferralRvrGpuTracegen::extension_opcodes())?;
        }
        ownership.validate_executed(replay_plan.executed_opcodes())?;
        let extension_opcodes = ownership.extension_opcodes();
        let rv64 = Rv64ImRvrGpuTracegen::new_after_claiming_extension_opcodes(
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
        mut self,
        vm: &mut VirtualMachine<BabyBearPoseidon2GpuEngine, SdkVmGpuBuilder>,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        let ctx = vm.generate_proving_ctx_from_rvr_unchecked_coverage(
            self.program,
            self.transcript,
            self.replay_plan,
            |_insertion_idx, chip| {
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
                    .generate_for_chip(_insertion_idx, chip)
                    .map_err(extension_error)
            },
        )?;
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
        self.rv64.finish().map_err(extension_error)?;
        vm.complete_rvr_tracegen_session();
        Ok(ctx)
    }
}

fn extension_error(error: GpuRvrInputError) -> GenerationError {
    GenerationError::ExtensionTracegen(error.to_string())
}

fn validate_checkpoint_config(
    has_modular: bool,
    has_fp2: bool,
    has_ecc: bool,
) -> Result<(), GpuRvrInputError> {
    if !has_modular && has_fp2 {
        return Err(GpuRvrInputError::InvalidAccessSchedule(
            "Fp2 checkpoint replay requires the Modular extension".to_string(),
        ));
    }
    if !has_modular && has_ecc {
        return Err(GpuRvrInputError::InvalidAccessSchedule(
            "Weierstrass checkpoint replay requires the Modular extension".to_string(),
        ));
    }
    Ok(())
}

struct OpcodeOwnership {
    native: RvrCheckpointOpcodeBases,
    extensions: BTreeMap<u32, &'static str>,
}

impl OpcodeOwnership {
    fn new(native: RvrCheckpointOpcodeBases) -> Self {
        Self {
            native,
            extensions: BTreeMap::new(),
        }
    }

    fn claim(
        &mut self,
        owner: &'static str,
        opcodes: impl IntoIterator<Item = u32>,
    ) -> Result<(), GpuRvrInputError> {
        for opcode in opcodes {
            if self.native.owns(opcode) {
                return Err(GpuRvrInputError::InvalidTranscript(format!(
                    "{owner} opcode {opcode:#x} collides with RV64/system"
                )));
            }
            if let Some(previous) = self.extensions.get(&opcode) {
                return Err(GpuRvrInputError::InvalidTranscript(format!(
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
    ) -> Result<(), GpuRvrInputError> {
        if let Some(opcode) = executed
            .into_iter()
            .find(|opcode| !self.native.owns(*opcode) && !self.extensions.contains_key(opcode))
        {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "executed opcode {opcode:#x} has no standard SDK RVR trace producer"
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
    use super::*;

    #[test]
    fn opcode_ownership_rejects_duplicates_and_missing_producers() {
        let native = Rv64ImRvrGpuTracegen::checkpoint_opcode_bases();
        let mut free_opcodes = (0..=u16::MAX as u32).filter(|opcode| !native.owns(*opcode));
        let claimed_opcode = free_opcodes.next().unwrap();
        let missing_opcode = free_opcodes.next().unwrap();
        let mut ownership = OpcodeOwnership::new(native);
        ownership.claim("first", [claimed_opcode]).unwrap();

        let duplicate = ownership.claim("second", [claimed_opcode]).unwrap_err();
        assert!(duplicate.to_string().contains("owned by both"));

        let native_collision = ownership.claim("extension", [native.phantom]).unwrap_err();
        assert!(native_collision.to_string().contains("RV64/system"));

        let missing = ownership.validate_executed([missing_opcode]).unwrap_err();
        assert!(missing
            .to_string()
            .contains("has no standard SDK RVR trace producer"));

        ownership
            .validate_executed([native.terminate, claimed_opcode])
            .unwrap();
    }
}
