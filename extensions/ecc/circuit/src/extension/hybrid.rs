//! GPU prover extension. Preflight replay uses native GPU trace generation for recognized
//! fields and a CPU postflight projection for other field expressions.

use std::{any::Any, collections::BTreeSet, sync::Arc};

use openvm_algebra_circuit::{
    cuda::field_expr::FieldExprReplayChip, AlgebraPreflightGpuTracegen, Fp2Extension,
    ModularExtension, Rv64ModularHybridBuilder,
};
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
use openvm_ecc_transpiler::Rv64WeierstrassOpcode;
use openvm_instructions::{program::Program, LocalOpcode};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_riscv_circuit::Rv64ImPreflightGpuTracegen;
use openvm_stark_backend::prover::{AirProvingContext, ProvingContext};
use strum::EnumCount;
#[cfg(feature = "rvr")]
use {
    crate::CurveConfig,
    openvm_circuit::arch::rvr::cuda::{
        PostflightAccessRegistry, PostflightAccessSchedule, PostflightAccessSpan,
    },
};
#[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
use {
    openvm_circuit::arch::rvr::{cuda::CheckpointReplayProgram, PreflightExecution},
    openvm_stark_backend::p3_field::PrimeField32,
};

use crate::{
    get_ec_addne_chip, get_ec_double_chip,
    weierstrass_chip::{
        generate_add_ne_trace_from_postflight, generate_double_trace_from_postflight,
    },
    Rv64WeierstrassConfig, WeierstrassAir, WeierstrassChip, WeierstrassExtension, ECC_BLOCKS_32,
    ECC_BLOCKS_48, NUM_LIMBS_32, NUM_LIMBS_48,
};

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
    ) -> Self {
        let replay = FieldExprReplayChip::new(&cpu, opcode_base, range_checker)
            .expect("valid Weierstrass field-expression replay configuration");
        Self {
            cpu,
            device_ctx,
            replay: Some(replay),
        }
    }

    fn local_opcodes() -> Result<[usize; 2], GpuPostflightError> {
        match NUM_READS {
            2 => Ok([
                Rv64WeierstrassOpcode::EC_ADD_NE as usize,
                Rv64WeierstrassOpcode::SETUP_EC_ADD_NE as usize,
            ]),
            1 => Ok([
                Rv64WeierstrassOpcode::EC_DOUBLE as usize,
                Rv64WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
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
            let opcode_base = Rv64WeierstrassOpcode::CLASS_OFFSET
                .checked_add(
                    curve_idx
                        .checked_mul(Rv64WeierstrassOpcode::COUNT)
                        .ok_or_else(|| {
                            GpuPostflightError::InvalidAccessSchedule(
                                "Weierstrass opcode range overflow".to_string(),
                            )
                        })?,
                )
                .ok_or_else(|| {
                    GpuPostflightError::InvalidAccessSchedule(
                        "Weierstrass opcode range overflow".to_string(),
                    )
                })?;
            let opcode = |local: Rv64WeierstrassOpcode| {
                let opcode = opcode_base.checked_add(local as usize).ok_or_else(|| {
                    GpuPostflightError::InvalidAccessSchedule(
                        "Weierstrass opcode range overflow".to_string(),
                    )
                })?;
                u32::try_from(opcode).map_err(|_| GpuPostflightError::OpcodeTooLarge(opcode))
            };
            let add_spans = [
                PostflightAccessSpan::read_fixed(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    0,
                    blocks as u32,
                ),
                PostflightAccessSpan::read_fixed(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    1,
                    blocks as u32,
                ),
                PostflightAccessSpan::write_fixed_from_replay_values(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
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
                Rv64WeierstrassOpcode::EC_ADD_NE,
                Rv64WeierstrassOpcode::SETUP_EC_ADD_NE,
            ] {
                registry.register(opcode(local)?, add_schedule)?;
            }
            let double_spans = [
                PostflightAccessSpan::read_fixed(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    0,
                    blocks as u32,
                ),
                PostflightAccessSpan::write_fixed_from_replay_values(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
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
            registry.register(opcode(Rv64WeierstrassOpcode::EC_DOUBLE)?, double_schedule)?;
            let setup_words = ec_double_setup_words(
                curve,
                blocks * openvm_circuit::arch::MEMORY_BLOCK_BYTES / 2,
            )?;
            let setup_double_spans = [
                PostflightAccessSpan::read_fixed(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    0,
                    blocks as u32,
                ),
                registry.write_fixed_from_static(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    1,
                    &setup_words,
                )?,
            ];
            registry.register(
                opcode(Rv64WeierstrassOpcode::SETUP_EC_DOUBLE)?,
                PostflightAccessSchedule {
                    spans: &setup_double_spans,
                    ..double_schedule
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
    ) -> Result<CheckpointReplayProgram, GpuPostflightError> {
        let mut registry = PostflightAccessRegistry::default();
        AlgebraPreflightGpuTracegen::register_postflight_access_schedules(
            &mut registry,
            modular,
            fp2,
        )?;
        Self::register_postflight_access_schedules(&mut registry, weierstrass)?;
        registry
            .validate_no_native_collisions(Rv64ImPreflightGpuTracegen::postflight_opcode_bases())?;
        CheckpointReplayProgram::upload_with_postflight_access_registry(
            program,
            memory_config,
            &registry,
            device_ctx,
        )
    }

    #[cfg(all(feature = "rvr", any(test, feature = "test-utils")))]
    pub fn postflight<VB>(
        vm: &VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
        program: &CheckpointReplayProgram,
        execution: &PreflightExecution,
        num_insns: u32,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError>
    where
        VB: VmBuilder<GpuBabyBearPoseidon2Engine, SystemChipInventory = SystemChipInventoryGPU>,
    {
        vm.postflight(
            program,
            execution,
            num_insns,
            Rv64ImPreflightGpuTracegen::postflight_opcode_bases(),
        )
    }

    pub fn new(
        extension: &WeierstrassExtension,
        program: &'a GpuPostflightProgram,
        transcript: &'a GpuPostflightTranscript,
        replay_plan: &'a GpuPostflightPlan,
    ) -> Self {
        let claimed_opcodes = (0..extension.supported_curves.len())
            .flat_map(|curve_idx| {
                let base =
                    Rv64WeierstrassOpcode::CLASS_OFFSET + curve_idx * Rv64WeierstrassOpcode::COUNT;
                (0..Rv64WeierstrassOpcode::COUNT).map(move |local| (base + local) as u32)
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
        }
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

    /// Returns `Some` only for a Weierstrass-owned AIR, allowing a concrete
    /// combined coordinator to fall through to algebra and RV64 producers.
    pub fn generate_for_chip(
        &mut self,
        chip: &dyn Any,
    ) -> Result<Option<AirProvingContext<GpuBackend>>, GpuPostflightError> {
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
                Rv64WeierstrassOpcode::CLASS_OFFSET + curve_idx * Rv64WeierstrassOpcode::COUNT;

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
                );
                inventory.add_postflight_executor_chip(addne, move |chip, postflight| {
                    let trace =
                        generate_add_ne_trace_from_postflight(&chip.cpu, postflight, opcode_base)?;
                    Ok(cpu_proving_ctx_to_gpu(
                        AirProvingContext::simple_no_pis(trace),
                        &chip.device_ctx,
                    ))
                });

                inventory.next_air::<WeierstrassAir<1, ECC_BLOCKS_32>>()?;
                let double = get_ec_double_chip::<F, ECC_BLOCKS_32>(
                    config,
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
                );
                inventory.add_postflight_executor_chip(double, move |chip, postflight| {
                    let trace =
                        generate_double_trace_from_postflight(&chip.cpu, postflight, opcode_base)?;
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
                );
                inventory.add_postflight_executor_chip(addne, move |chip, postflight| {
                    let trace =
                        generate_add_ne_trace_from_postflight(&chip.cpu, postflight, opcode_base)?;
                    Ok(cpu_proving_ctx_to_gpu(
                        AirProvingContext::simple_no_pis(trace),
                        &chip.device_ctx,
                    ))
                });

                inventory.next_air::<WeierstrassAir<1, ECC_BLOCKS_48>>()?;
                let double = get_ec_double_chip::<F, ECC_BLOCKS_48>(
                    config,
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
                );
                inventory.add_postflight_executor_chip(double, move |chip, postflight| {
                    let trace =
                        generate_double_trace_from_postflight(&chip.cpu, postflight, opcode_base)?;
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
        _host_program: &Program<F>,
        program: &Self::Prepared,
        output: &PreflightOutput,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError> {
        let (transcript, replay_plan) = vm
            .postflight_history(program, output)
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        let config = vm.config().clone();
        WeierstrassPreflightGpuTracegen::new(
            &config.weierstrass,
            program,
            &transcript,
            &replay_plan,
        )
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
