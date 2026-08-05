//! GPU prover extension. Preflight replay uses native GPU trace generation for recognized
//! fields and a CPU postflight projection for other field expressions.

use std::{any::Any, collections::BTreeSet, sync::Arc};

use openvm_algebra_circuit::{
    cuda::field_expr::FieldExprReplayChip, AlgebraPreflightGpuTracegen, Fp2Extension,
    ModularExtension, Rv64ModularHybridBuilder,
};
#[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
use openvm_circuit::arch::rvr::PreflightExecution;
use openvm_circuit::{
    arch::{
        cuda::postflight::{
            GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram, GpuPostflightTranscript,
        },
        GenerationError, VirtualMachine, *,
    },
    system::{
        cuda::{extensions::get_inventory_range_checker, SystemChipInventoryGPU},
        memory::SharedMemoryHelper,
    },
};
use openvm_circuit_primitives::{
    hybrid_chip::cpu_proving_ctx_to_gpu, var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{
    prelude::{F, SC},
    BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend,
};
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_ecc_transpiler::WeierstrassOpcode;
use openvm_instructions::{program::Program, LocalOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
#[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
use openvm_riscv_circuit::preflight::PreflightReplayProgram;
#[cfg(feature = "rvr")]
use openvm_riscv_circuit::preflight::{
    PostflightAccessRegistry, PostflightAccessSchedule, PostflightAccessSpan,
};
use openvm_riscv_circuit::Rv64ImPreflightGpuTracegen;
#[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_backend::prover::{AirProvingContext, ProvingContext};
use strum::EnumCount;

use crate::{
    generate_ec_mul_trace_from_postflight, get_ec_addne_chip, get_ec_double_chip, get_ec_mul_chip,
    weierstrass_chip::{
        generate_add_ne_trace_from_postflight, generate_double_trace_from_postflight,
    },
    EcMulAir, EcMulChip, Rv64WeierstrassConfig, WeierstrassAir, WeierstrassChip,
    WeierstrassExtension, ECC_BLOCKS_32, ECC_BLOCKS_48, NUM_LIMBS_32, NUM_LIMBS_48,
};
#[cfg(feature = "rvr")]
use crate::{CurveConfig, SCALAR_BLOCKS};

#[cfg(feature = "rvr")]
fn ec_double_setup_words(
    curve: &CurveConfig,
    coordinate_bytes: usize,
) -> Result<Vec<u64>, GpuPostflightError> {
    if curve.modulus == Default::default()
        || curve.a >= curve.modulus
        || curve.modulus.bits().div_ceil(8) as usize > coordinate_bytes
        || !coordinate_bytes.is_multiple_of(std::mem::size_of::<u64>())
    {
        return Err(GpuPostflightError::InvalidAccessSchedule(format!(
            "invalid setup constants for curve {}",
            curve.struct_name
        )));
    }
    // In a setup row the expression inputs are x1 = p and y1 = a. Modulo p,
    // lambda = a, so the fixed postimage is (a^2, -a^3-a).
    let x = (&curve.a * &curve.a) % &curve.modulus;
    let neg_y = ((&x * &curve.a) + &curve.a) % &curve.modulus;
    let y = if neg_y == Default::default() {
        Default::default()
    } else {
        &curve.modulus - neg_y
    };
    let mut bytes = vec![0u8; 2 * coordinate_bytes];
    for (index, value) in [x, y].iter().enumerate() {
        let value = value.to_bytes_le();
        bytes[index * coordinate_bytes..index * coordinate_bytes + value.len()]
            .copy_from_slice(&value);
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u64>())
        .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
        .collect())
}

#[cfg(all(test, feature = "rvr"))]
mod checkpoint_tests {
    use super::*;
    use crate::{P256_CONFIG, SECP256K1_CONFIG};

    #[test]
    fn ec_double_setup_words_match_configured_expression() {
        assert_eq!(
            ec_double_setup_words(&P256_CONFIG, NUM_LIMBS_32).unwrap(),
            [9, 0, 0, 0, 30, 0, 0, 0]
        );
        assert_eq!(
            ec_double_setup_words(&SECP256K1_CONFIG, NUM_LIMBS_32).unwrap(),
            [0; ECC_BLOCKS_32]
        );
    }
}

pub struct HybridWeierstrassChip<F, const NUM_READS: usize, const BLOCKS: usize> {
    cpu: WeierstrassChip<F, NUM_READS, BLOCKS>,
    device_ctx: GpuDeviceCtx,
    replay: Option<FieldExprReplayChip<NUM_READS, BLOCKS>>,
}

impl<const NUM_READS: usize, const BLOCKS: usize> HybridWeierstrassChip<F, NUM_READS, BLOCKS> {
    pub fn new(cpu: WeierstrassChip<F, NUM_READS, BLOCKS>, device_ctx: GpuDeviceCtx) -> Self {
        Self {
            cpu,
            device_ctx,
            replay: None,
        }
    }

    pub fn new_with_replay(
        cpu: WeierstrassChip<F, NUM_READS, BLOCKS>,
        device_ctx: GpuDeviceCtx,
        opcode_base: usize,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
    ) -> Result<Self, GpuPostflightError> {
        let replay = FieldExprReplayChip::new(&cpu, opcode_base, range_checker)?;
        Ok(Self {
            cpu,
            device_ctx,
            replay: Some(replay),
        })
    }

    fn local_opcodes() -> Result<[usize; 2], GpuPostflightError> {
        match NUM_READS {
            2 => Ok([
                WeierstrassOpcode::EC_ADD_NE as usize,
                WeierstrassOpcode::SETUP_EC_ADD_NE as usize,
            ]),
            1 => Ok([
                WeierstrassOpcode::EC_DOUBLE as usize,
                WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
            ]),
            _ => Err(GpuPostflightError::InvalidTranscript(format!(
                "unsupported Weierstrass replay read count {NUM_READS}"
            ))),
        }
    }

    pub fn opcode_base(&self) -> Option<usize> {
        self.replay.as_ref().map(|replay| replay.opcode_base())
    }

    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let replay = self.replay.as_ref().ok_or_else(|| {
            GpuPostflightError::InvalidTranscript(
                "Weierstrass chip was constructed without postflight replay".to_string(),
            )
        })?;
        replay.generate_proving_ctx(&self.cpu, program, transcript, replay_plan)
    }
}

/// GPU-side wrapper for the CPU `EC_MUL` chip.
///
/// Unlike the add-ne and double chips, this one has no [`FieldExprReplayChip`]. That replay path is
/// typed for a [`WeierstrassChip`], and its projection kernel accepts only a fixed set of
/// (reads, blocks) shapes, none of which describe `EC_MUL`'s point + 256-bit scalar + point access
/// pattern or its multirow trace. The trace is therefore built on the CPU and uploaded.
pub struct HybridEcMulChip<F, const NUM_LIMBS: usize, const BLOCKS: usize> {
    cpu: EcMulChip<F, NUM_LIMBS, BLOCKS>,
    device_ctx: GpuDeviceCtx,
    opcode_base: usize,
}

impl<const NUM_LIMBS: usize, const BLOCKS: usize> HybridEcMulChip<F, NUM_LIMBS, BLOCKS> {
    pub fn new(
        cpu: EcMulChip<F, NUM_LIMBS, BLOCKS>,
        device_ctx: GpuDeviceCtx,
        opcode_base: usize,
    ) -> Self {
        Self {
            cpu,
            device_ctx,
            opcode_base,
        }
    }

    fn local_opcodes() -> [usize; 2] {
        [
            WeierstrassOpcode::EC_MUL as usize,
            WeierstrassOpcode::SETUP_EC_MUL as usize,
        ]
    }
}

/// Prover extension for hybrid CPU trace generation and GPU proving.
#[derive(Clone, Copy, Default)]
pub struct EccHybridProverExt;

/// Concrete inventory visitor for Weierstrass postflight trace generation.
///
/// Multiple configured curves may have the same Rust chip type, so coverage is
/// tracked by each chip's concrete opcode base rather than by downcast type.
pub struct WeierstrassPreflightGpuTracegen<'a> {
    program: &'a GpuPostflightProgram,
    transcript: &'a GpuPostflightTranscript,
    replay_plan: &'a GpuPostflightPlan,
    claimed_opcodes: Vec<u32>,
    pending_opcodes: BTreeSet<u32>,
    /// Host postflight for the `EC_MUL` chip, whose multirow trace has no GPU projection.
    cpu_postflight: Option<&'a Postflight<'a, F>>,
}

impl<'a> WeierstrassPreflightGpuTracegen<'a> {
    #[cfg(feature = "rvr")]
    #[doc(hidden)]
    pub fn register_postflight_access_schedules(
        registry: &mut PostflightAccessRegistry,
        extension: &WeierstrassExtension,
    ) -> Result<(), GpuPostflightError> {
        for (curve_idx, curve) in extension.supported_curves.iter().enumerate() {
            let bytes = curve.modulus.bits().div_ceil(8) as usize;
            let blocks = if bytes <= NUM_LIMBS_32 {
                ECC_BLOCKS_32
            } else if bytes <= NUM_LIMBS_48 {
                ECC_BLOCKS_48
            } else {
                return Err(GpuPostflightError::InvalidAccessSchedule(format!(
                    "Weierstrass curve {curve_idx} exceeds the supported 48-byte layout"
                )));
            };
            let opcode_base =
                WeierstrassOpcode::CLASS_OFFSET
                    .checked_add(curve_idx.checked_mul(WeierstrassOpcode::COUNT).ok_or_else(
                        || {
                            GpuPostflightError::InvalidAccessSchedule(
                                "Weierstrass opcode range overflow".to_string(),
                            )
                        },
                    )?)
                    .ok_or_else(|| {
                        GpuPostflightError::InvalidAccessSchedule(
                            "Weierstrass opcode range overflow".to_string(),
                        )
                    })?;
            let opcode = |local: WeierstrassOpcode| {
                let opcode = opcode_base.checked_add(local as usize).ok_or_else(|| {
                    GpuPostflightError::InvalidAccessSchedule(
                        "Weierstrass opcode range overflow".to_string(),
                    )
                })?;
                u32::try_from(opcode).map_err(|_| GpuPostflightError::OpcodeTooLarge(opcode))
            };
            let add_spans = [
                PostflightAccessSpan::read_fixed(
                    openvm_instructions::riscv::MEMORY_AS,
                    0,
                    blocks as u32,
                ),
                PostflightAccessSpan::read_fixed(
                    openvm_instructions::riscv::MEMORY_AS,
                    1,
                    blocks as u32,
                ),
                PostflightAccessSpan::write_fixed_from_replay_values(
                    openvm_instructions::riscv::MEMORY_AS,
                    2,
                    blocks as u32,
                ),
            ];
            let add_schedule = PostflightAccessSchedule {
                register_operands: &[2, 3, 1],
                zero_operand_mask: (1 << 6) | (1 << 7),
                register_as_operand: 4,
                memory_as_operand: 5,
                spans: &add_spans,
            };
            for local in [
                WeierstrassOpcode::EC_ADD_NE,
                WeierstrassOpcode::SETUP_EC_ADD_NE,
            ] {
                registry.register(opcode(local)?, add_schedule)?;
            }
            let double_spans = [
                PostflightAccessSpan::read_fixed(
                    openvm_instructions::riscv::MEMORY_AS,
                    0,
                    blocks as u32,
                ),
                PostflightAccessSpan::write_fixed_from_replay_values(
                    openvm_instructions::riscv::MEMORY_AS,
                    1,
                    blocks as u32,
                ),
            ];
            let double_schedule = PostflightAccessSchedule {
                register_operands: &[2, 1],
                zero_operand_mask: (1 << 3) | (1 << 6) | (1 << 7),
                register_as_operand: 4,
                memory_as_operand: 5,
                spans: &double_spans,
            };
            registry.register(opcode(WeierstrassOpcode::EC_DOUBLE)?, double_schedule)?;
            let setup_words = ec_double_setup_words(
                curve,
                blocks * openvm_circuit::arch::MEMORY_BLOCK_BYTES / 2,
            )?;
            let setup_double_spans = [
                PostflightAccessSpan::read_fixed(
                    openvm_instructions::riscv::MEMORY_AS,
                    0,
                    blocks as u32,
                ),
                registry.write_fixed_from_static(
                    openvm_instructions::riscv::MEMORY_AS,
                    1,
                    &setup_words,
                )?,
            ];
            registry.register(
                opcode(WeierstrassOpcode::SETUP_EC_DOUBLE)?,
                PostflightAccessSchedule {
                    spans: &setup_double_spans,
                    ..double_schedule
                },
            )?;

            // `EC_MUL` reads a point and a fixed-width scalar and writes a point. The scalar is
            // read on setup rows too, so both schedules share the read spans.
            let mul_spans = [
                PostflightAccessSpan::read_fixed(
                    openvm_instructions::riscv::MEMORY_AS,
                    0,
                    blocks as u32,
                ),
                PostflightAccessSpan::read_fixed(
                    openvm_instructions::riscv::MEMORY_AS,
                    1,
                    SCALAR_BLOCKS as u32,
                ),
                PostflightAccessSpan::write_fixed_from_replay_values(
                    openvm_instructions::riscv::MEMORY_AS,
                    2,
                    blocks as u32,
                ),
            ];
            let mul_schedule = PostflightAccessSchedule {
                register_operands: &[2, 3, 1],
                zero_operand_mask: (1 << 6) | (1 << 7),
                register_as_operand: 4,
                memory_as_operand: 5,
                spans: &mul_spans,
            };
            registry.register(opcode(WeierstrassOpcode::EC_MUL)?, mul_schedule)?;

            // A setup row sets no case flag, so the output selects fall through to the base point,
            // which the setup inputs leave zero. `ec_mul_setup_postimage_is_zero` in the rvr FFI
            // crate pins this for every curve.
            let setup_mul_words = vec![0u64; blocks];
            let setup_mul_spans = [
                mul_spans[0],
                mul_spans[1],
                registry.write_fixed_from_static(
                    openvm_instructions::riscv::MEMORY_AS,
                    2,
                    &setup_mul_words,
                )?,
            ];
            registry.register(
                opcode(WeierstrassOpcode::SETUP_EC_MUL)?,
                PostflightAccessSchedule {
                    spans: &setup_mul_spans,
                    ..mul_schedule
                },
            )?;
        }
        Ok(())
    }

    /// Uploads one concrete RV64+Algebra+Weierstrass checkpoint program.
    #[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
    pub fn upload_postflight_program<T: PrimeField32>(
        program: &Program<T>,
        memory_config: &MemoryConfig,
        modular: &ModularExtension,
        fp2: Option<&Fp2Extension>,
        weierstrass: &WeierstrassExtension,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<PreflightReplayProgram, GpuPostflightError> {
        let mut registry = PostflightAccessRegistry::default();
        AlgebraPreflightGpuTracegen::register_postflight_access_schedules(
            &mut registry,
            modular,
            fp2,
        )?;
        Self::register_postflight_access_schedules(&mut registry, weierstrass)?;
        PreflightReplayProgram::upload_with_postflight_access_registry(
            program,
            memory_config,
            &registry,
            device_ctx,
        )
    }

    #[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
    pub fn postflight<VB>(
        vm: &VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
        program: &PreflightReplayProgram,
        execution: &PreflightExecution,
        num_insns: u32,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError>
    where
        VB: VmBuilder<GpuBabyBearPoseidon2Engine, SystemChipInventory = SystemChipInventoryGPU>,
    {
        Rv64ImPreflightGpuTracegen::postflight(vm, program, execution, num_insns)
    }

    pub fn new(
        extension: &WeierstrassExtension,
        program: &'a GpuPostflightProgram,
        transcript: &'a GpuPostflightTranscript,
        replay_plan: &'a GpuPostflightPlan,
    ) -> Self {
        let claimed_opcodes = (0..extension.supported_curves.len())
            .flat_map(|curve_idx| {
                let base = WeierstrassOpcode::CLASS_OFFSET + curve_idx * WeierstrassOpcode::COUNT;
                (0..WeierstrassOpcode::COUNT).map(move |local| (base + local) as u32)
            })
            .collect::<Vec<_>>();
        let pending_opcodes = claimed_opcodes
            .iter()
            .copied()
            .filter(|&opcode| {
                !replay_plan
                    .opcode_range(openvm_instructions::VmOpcode::from_usize(opcode as usize))
                    .is_empty()
            })
            .collect();
        Self {
            program,
            transcript,
            replay_plan,
            claimed_opcodes,
            pending_opcodes,
            cpu_postflight: None,
        }
    }

    /// Supplies the host postflight used to generate the `EC_MUL` trace.
    ///
    /// Required whenever the segment executes `EC_MUL` or `SETUP_EC_MUL`. Callers that own the
    /// interpreter's preflight output can always provide it; those working only from a GPU
    /// transcript cannot, and such a segment is rejected rather than mis-traced.
    #[must_use]
    pub fn with_cpu_postflight(mut self, postflight: &'a Postflight<'a, F>) -> Self {
        self.cpu_postflight = Some(postflight);
        self
    }

    pub fn claimed_opcodes(&self) -> &[u32] {
        &self.claimed_opcodes
    }

    fn generate_for_weierstrass_chip<const NUM_READS: usize, const BLOCKS: usize>(
        &mut self,
        chip: &HybridWeierstrassChip<F, NUM_READS, BLOCKS>,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let base = chip.opcode_base().ok_or_else(|| {
            GpuPostflightError::InvalidTranscript(
                "Weierstrass inventory chip has no postflight replay configuration".to_string(),
            )
        })?;
        for local in HybridWeierstrassChip::<F, NUM_READS, BLOCKS>::local_opcodes()? {
            let opcode = u32::try_from(base + local)
                .map_err(|_| GpuPostflightError::OpcodeTooLarge(base + local))?;
            self.pending_opcodes.remove(&opcode);
        }
        chip.generate_proving_ctx_from_postflight(self.program, self.transcript, self.replay_plan)
    }

    /// `EC_MUL`'s trace is built on the host and uploaded.
    ///
    /// The shared vec-heap gather kernel accepts only a fixed set of (reads, blocks) shapes with
    /// uniform read widths, and none describes this chip's point + 256-bit scalar + point schedule
    /// or its multirow trace. Rather than add a kernel, this reuses the same host trace generator
    /// the CPU prover uses, so both backends produce identical traces by construction.
    fn generate_for_ec_mul_chip<const NUM_LIMBS: usize, const BLOCKS: usize>(
        &mut self,
        chip: &HybridEcMulChip<F, NUM_LIMBS, BLOCKS>,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let mut used = 0usize;
        for local in HybridEcMulChip::<F, NUM_LIMBS, BLOCKS>::local_opcodes() {
            let opcode = chip.opcode_base + local;
            let opcode =
                u32::try_from(opcode).map_err(|_| GpuPostflightError::OpcodeTooLarge(opcode))?;
            used += self
                .replay_plan
                .opcode_range(openvm_instructions::VmOpcode::from_usize(opcode as usize))
                .len();
            self.pending_opcodes.remove(&opcode);
        }

        let Some(postflight) = self.cpu_postflight else {
            // Every configured curve registers an `EC_MUL` chip whether or not the program uses
            // one, so an unused chip must still succeed here.
            if used == 0 {
                return Ok(AirProvingContext::simple_no_pis(
                    openvm_cuda_backend::base::DeviceMatrix::dummy(),
                ));
            }
            // `sw_declare!`'s one-time setup emits SETUP_EC_MUL for every declared curve, so any
            // program touching a Weierstrass curve reaches this even without a multiplication.
            return Err(GpuPostflightError::InvalidTranscript(format!(
                "EC_MUL trace generation needs the host postflight, which this caller did not \
                 supply; {used} instruction(s) at opcode base {}",
                chip.opcode_base
            )));
        };

        let trace = generate_ec_mul_trace_from_postflight(&chip.cpu, postflight, chip.opcode_base)
            .map_err(|error| {
                GpuPostflightError::InvalidTranscript(format!(
                    "EC_MUL host trace generation failed: {error:?}"
                ))
            })?;
        Ok(cpu_proving_ctx_to_gpu(
            AirProvingContext::simple_no_pis(trace),
            &chip.device_ctx,
        ))
    }

    /// Returns `Some` only for a Weierstrass-owned AIR, allowing a concrete
    /// combined coordinator to fall through to algebra and RV64 producers.
    pub fn generate_for_chip(
        &mut self,
        chip: &dyn Any,
    ) -> Result<Option<AirProvingContext<GpuBackend>>, GpuPostflightError> {
        if let Some(chip) = chip.downcast_ref::<HybridEcMulChip<F, NUM_LIMBS_32, ECC_BLOCKS_32>>() {
            return self.generate_for_ec_mul_chip(chip).map(Some);
        }
        if let Some(chip) = chip.downcast_ref::<HybridEcMulChip<F, NUM_LIMBS_48, ECC_BLOCKS_48>>() {
            return self.generate_for_ec_mul_chip(chip).map(Some);
        }
        if let Some(chip) = chip.downcast_ref::<HybridWeierstrassChip<F, 2, ECC_BLOCKS_32>>() {
            return self.generate_for_weierstrass_chip(chip).map(Some);
        }
        if let Some(chip) = chip.downcast_ref::<HybridWeierstrassChip<F, 1, ECC_BLOCKS_32>>() {
            return self.generate_for_weierstrass_chip(chip).map(Some);
        }
        if let Some(chip) = chip.downcast_ref::<HybridWeierstrassChip<F, 2, ECC_BLOCKS_48>>() {
            return self.generate_for_weierstrass_chip(chip).map(Some);
        }
        if let Some(chip) = chip.downcast_ref::<HybridWeierstrassChip<F, 1, ECC_BLOCKS_48>>() {
            return self.generate_for_weierstrass_chip(chip).map(Some);
        }
        Ok(None)
    }

    pub fn finish(self) -> Result<(), GpuPostflightError> {
        if self.pending_opcodes.is_empty() {
            Ok(())
        } else {
            Err(GpuPostflightError::InvalidTranscript(format!(
                "Weierstrass preflight GPU tracegen did not visit opcodes {:?}",
                self.pending_opcodes
            )))
        }
    }

    /// Generates the complete RISC-V + modular + optional Fp2 + Weierstrass proving context
    /// from one preflight execution.
    pub fn generate_proving_ctx<VB>(
        self,
        vm: &mut VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
        modular: &ModularExtension,
        fp2: Option<&Fp2Extension>,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError>
    where
        VB: VmBuilder<GpuBabyBearPoseidon2Engine, SystemChipInventory = SystemChipInventoryGPU>,
    {
        let algebra = AlgebraPreflightGpuTracegen::new(
            self.program,
            self.transcript,
            self.replay_plan,
            modular,
            fp2,
        )
        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        let mut extension_opcodes = self.claimed_opcodes.clone();
        extension_opcodes.extend_from_slice(algebra.extension_opcodes());
        let rv64 = Rv64ImPreflightGpuTracegen::new_after_claiming_extension_opcodes(
            self.program,
            self.transcript,
            self.replay_plan,
            &extension_opcodes,
        )
        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        vm.generate_preflight_proving_ctx(
            self.program,
            self.transcript,
            self.replay_plan,
            (self, algebra, rv64),
            |(tracegen, algebra, rv64), chip| {
                if let Some(ctx) = tracegen
                    .generate_for_chip(chip)
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?
                {
                    Ok(ctx)
                } else if let Some(ctx) = algebra
                    .generate_for_chip(chip)
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?
                {
                    Ok(ctx)
                } else {
                    rv64.generate_for_chip(chip)
                        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
                }
            },
            |(tracegen, algebra, rv64)| {
                rv64.finish()
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
                algebra
                    .finish()
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
                tracegen
                    .finish()
                    .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
            },
        )
    }
}

impl VmProverExtension<GpuBabyBearPoseidon2Engine, WeierstrassExtension> for EccHybridProverExt {
    fn extend_prover(
        &self,
        extension: &WeierstrassExtension,
        inventory: &mut ChipInventory<SC, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker_gpu = get_inventory_range_checker(inventory);
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let range_checker = range_checker_gpu.cpu_chip.clone().unwrap();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

        let device_ctx = range_checker_gpu.device_ctx.clone();

        for (curve_idx, curve) in extension.supported_curves.iter().enumerate() {
            let bytes = curve.modulus.bits().div_ceil(8) as usize;
            let opcode_base =
                WeierstrassOpcode::CLASS_OFFSET + curve_idx * WeierstrassOpcode::COUNT;

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, ECC_BLOCKS_32>>()?;
                let addne = get_ec_addne_chip::<F, ECC_BLOCKS_32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                let addne = HybridWeierstrassChip::new_with_replay(
                    addne,
                    device_ctx.clone(),
                    opcode_base,
                    range_checker_gpu.clone(),
                )
                .map_err(|source| {
                    ChipInventoryError::prover_chip_initialization(
                        "Weierstrass add-ne replay",
                        source,
                    )
                })?;
                inventory.add_executor_chip_with_tracegen(addne, move |chip, postflight| {
                    let trace =
                        generate_add_ne_trace_from_postflight(&chip.cpu, postflight, opcode_base)?;
                    Ok(cpu_proving_ctx_to_gpu(
                        AirProvingContext::simple_no_pis(trace),
                        &chip.device_ctx,
                    ))
                });

                inventory.next_air::<WeierstrassAir<1, ECC_BLOCKS_32>>()?;
                let double = get_ec_double_chip::<F, ECC_BLOCKS_32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                    curve.a.clone(),
                );
                let double = HybridWeierstrassChip::new_with_replay(
                    double,
                    device_ctx.clone(),
                    opcode_base,
                    range_checker_gpu.clone(),
                )
                .map_err(|source| {
                    ChipInventoryError::prover_chip_initialization(
                        "Weierstrass double replay",
                        source,
                    )
                })?;
                inventory.add_executor_chip_with_tracegen(double, move |chip, postflight| {
                    let trace =
                        generate_double_trace_from_postflight(&chip.cpu, postflight, opcode_base)?;
                    Ok(cpu_proving_ctx_to_gpu(
                        AirProvingContext::simple_no_pis(trace),
                        &chip.device_ctx,
                    ))
                });

                inventory.next_air::<EcMulAir<NUM_LIMBS_32, ECC_BLOCKS_32>>()?;
                let mul = get_ec_mul_chip::<F, NUM_LIMBS_32, ECC_BLOCKS_32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                    curve.a.clone(),
                );
                let mul = HybridEcMulChip::new(mul, device_ctx.clone(), opcode_base);
                inventory.add_executor_chip_with_tracegen(mul, move |chip, postflight| {
                    let trace =
                        generate_ec_mul_trace_from_postflight(&chip.cpu, postflight, opcode_base)?;
                    Ok(cpu_proving_ctx_to_gpu(
                        AirProvingContext::simple_no_pis(trace),
                        &chip.device_ctx,
                    ))
                });
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, ECC_BLOCKS_48>>()?;
                let addne = get_ec_addne_chip::<F, ECC_BLOCKS_48>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                let addne = HybridWeierstrassChip::new_with_replay(
                    addne,
                    device_ctx.clone(),
                    opcode_base,
                    range_checker_gpu.clone(),
                )
                .map_err(|source| {
                    ChipInventoryError::prover_chip_initialization(
                        "Weierstrass add-ne replay",
                        source,
                    )
                })?;
                inventory.add_executor_chip_with_tracegen(addne, move |chip, postflight| {
                    let trace =
                        generate_add_ne_trace_from_postflight(&chip.cpu, postflight, opcode_base)?;
                    Ok(cpu_proving_ctx_to_gpu(
                        AirProvingContext::simple_no_pis(trace),
                        &chip.device_ctx,
                    ))
                });

                inventory.next_air::<WeierstrassAir<1, ECC_BLOCKS_48>>()?;
                let double = get_ec_double_chip::<F, ECC_BLOCKS_48>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                    curve.a.clone(),
                );
                let double = HybridWeierstrassChip::new_with_replay(
                    double,
                    device_ctx.clone(),
                    opcode_base,
                    range_checker_gpu.clone(),
                )
                .map_err(|source| {
                    ChipInventoryError::prover_chip_initialization(
                        "Weierstrass double replay",
                        source,
                    )
                })?;
                inventory.add_executor_chip_with_tracegen(double, move |chip, postflight| {
                    let trace =
                        generate_double_trace_from_postflight(&chip.cpu, postflight, opcode_base)?;
                    Ok(cpu_proving_ctx_to_gpu(
                        AirProvingContext::simple_no_pis(trace),
                        &chip.device_ctx,
                    ))
                });

                inventory.next_air::<EcMulAir<NUM_LIMBS_48, ECC_BLOCKS_48>>()?;
                let mul = get_ec_mul_chip::<F, NUM_LIMBS_48, ECC_BLOCKS_48>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                    curve.a.clone(),
                );
                let mul = HybridEcMulChip::new(mul, device_ctx.clone(), opcode_base);
                inventory.add_executor_chip_with_tracegen(mul, move |chip, postflight| {
                    let trace =
                        generate_ec_mul_trace_from_postflight(&chip.cpu, postflight, opcode_base)?;
                    Ok(cpu_proving_ctx_to_gpu(
                        AirProvingContext::simple_no_pis(trace),
                        &chip.device_ctx,
                    ))
                });
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

/// GPU builder for RV64IM, modular, and elliptic-curve extensions.
#[derive(Clone)]
pub struct Rv64WeierstrassHybridBuilder;

impl PostflightTracegen<GpuBabyBearPoseidon2Engine> for Rv64WeierstrassHybridBuilder {
    type Prepared = GpuPostflightProgram;

    fn prepare_postflight(
        vm: &VirtualMachine<GpuBabyBearPoseidon2Engine, Self>,
        program: &Program<F>,
    ) -> Result<Self::Prepared, GenerationError> {
        prepare_gpu_postflight(vm, program)
    }

    fn generate_proving_ctx(
        vm: &mut VirtualMachine<GpuBabyBearPoseidon2Engine, Self>,
        host_program: &Program<F>,
        program: &Self::Prepared,
        output: &PreflightOutput,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        let (transcript, replay_plan) = vm
            .postflight_history(program, output)
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        let config = vm.config().clone();
        // `EC_MUL` has no GPU projection, so its trace comes from the host postflight.
        let memory_config = vm.config().as_ref().memory_config.clone();
        let cpu_postflight = Postflight::new(
            host_program,
            &output.history,
            &memory_config,
            output.exit_code,
        )
        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        WeierstrassPreflightGpuTracegen::new(
            &config.weierstrass,
            program,
            &transcript,
            &replay_plan,
        )
        .with_cpu_postflight(&cpu_postflight)
        .generate_proving_ctx(vm, &config.modular.modular, None)
    }
}

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Rv64WeierstrassHybridBuilder {
    type VmConfig = Rv64WeierstrassConfig;
    type SystemChipInventory = SystemChipInventoryGPU;

    fn create_chip_complex(
        &self,
        config: &Rv64WeierstrassConfig,
        circuit: AirInventory<SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<VmChipComplex<SC, GpuBackend, Self::SystemChipInventory>, ChipInventoryError> {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &Rv64ModularHybridBuilder,
            &config.modular,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _>::extend_prover(
            &EccHybridProverExt,
            &config.weierstrass,
            inventory,
        )?;

        Ok(chip_complex)
    }
}
