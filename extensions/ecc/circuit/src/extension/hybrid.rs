//! GPU prover extension. Record-based trace generation uses the CPU fallback. Preflight
//! replay uses record-free GPU trace generation for recognized fields and an arena-free CPU
//! projection for other field expressions.

use openvm_algebra_circuit::Rv64ModularHybridBuilder;
use openvm_circuit::{
    arch::*,
    system::{
        cuda::{extensions::get_inventory_range_checker, SystemChipInventoryGPU},
        memory::SharedMemoryHelper,
    },
};
use openvm_circuit_primitives::{hybrid_chip::cpu_proving_ctx_to_gpu, Chip};
use openvm_cpu_backend::CpuBackend;
use openvm_cuda_backend::{
    base::DeviceMatrix,
    prelude::{F, SC},
    BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend,
};
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_mod_circuit_builder::{ExprBuilderConfig, FieldExpressionMetadata};
use openvm_riscv_adapters::{Rv64VecHeapAdapterCols, Rv64VecHeapAdapterExecutor};
use openvm_stark_backend::{p3_air::BaseAir, prover::AirProvingContext};
#[cfg(feature = "rvr")]
use {
    openvm_algebra_circuit::{
        cuda::field_expr::FieldExprReplayChip, AlgebraPreflightGpuTracegen, Fp2Extension,
        ModularExtension,
    },
    openvm_circuit::arch::{
        rvr::{
            cuda::{
                GpuPostflightError, GpuPostflightPlan, GpuPostflightProgram,
                GpuPostflightTranscript, PostflightAccessRegistry, PostflightAccessSpan,
            },
            PreflightExecution,
        },
        GenerationError, VirtualMachine,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU,
    openvm_ecc_transpiler::Rv64WeierstrassOpcode,
    openvm_instructions::{program::Program, LocalOpcode},
    openvm_riscv_circuit::Rv64ImPreflightGpuTracegen,
    openvm_stark_backend::{p3_field::PrimeField32, prover::ProvingContext},
    std::{any::Any, collections::BTreeSet, sync::Arc},
    strum::EnumCount,
};

#[cfg(feature = "rvr")]
use crate::CurveConfig;
use crate::{
    get_ec_addne_chip, get_ec_double_chip, EccRecord, Rv64WeierstrassConfig, WeierstrassAir,
    WeierstrassChip, WeierstrassExtension, ECC_BLOCKS_32, ECC_BLOCKS_48, NUM_LIMBS_32,
    NUM_LIMBS_48,
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
    #[cfg(feature = "rvr")]
    replay: Option<FieldExprReplayChip<NUM_READS, BLOCKS>>,
}

impl<const NUM_READS: usize, const BLOCKS: usize> HybridWeierstrassChip<F, NUM_READS, BLOCKS> {
    pub fn new(cpu: WeierstrassChip<F, NUM_READS, BLOCKS>, device_ctx: GpuDeviceCtx) -> Self {
        Self {
            cpu,
            device_ctx,
            #[cfg(feature = "rvr")]
            replay: None,
        }
    }

    #[cfg(feature = "rvr")]
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

    #[cfg(feature = "rvr")]
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

    #[cfg(feature = "rvr")]
    pub fn opcode_base(&self) -> Option<usize> {
        self.replay.as_ref().map(|replay| replay.opcode_base())
    }

    #[cfg(feature = "rvr")]
    pub fn generate_proving_ctx_from_postflight(
        &self,
        program: &GpuPostflightProgram,
        transcript: &GpuPostflightTranscript,
        replay_plan: &GpuPostflightPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuPostflightError> {
        let replay = self.replay.as_ref().ok_or_else(|| {
            GpuPostflightError::InvalidTranscript(
                "Weierstrass chip was constructed without checkpoint replay".to_string(),
            )
        })?;
        replay.generate_proving_ctx(&self.cpu, program, transcript, replay_plan)
    }
}

// Auto-implementation of Chip for GpuBackend for a Cpu Chip by doing conversion
// of Dense->Matrix Record Arena, cpu tracegen, and then H2D transfer of the trace matrix.
impl<const NUM_READS: usize, const BLOCKS: usize> Chip<DenseRecordArena, GpuBackend>
    for HybridWeierstrassChip<F, NUM_READS, BLOCKS>
{
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.program().canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv64VecHeapAdapterExecutor<NUM_READS, BLOCKS, BLOCKS>,
        >::new(total_input_limbs));

        let record_size = RecordSeeker::<
            DenseRecordArena,
            EccRecord<NUM_READS, BLOCKS>,
            _,
        >::get_aligned_record_size(&layout);

        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let height = num_records.next_power_of_two();
        let mut seeker = arena.get_record_seeker::<EccRecord<NUM_READS, BLOCKS>, AdapterCoreLayout<
            FieldExpressionMetadata<F, Rv64VecHeapAdapterExecutor<NUM_READS, BLOCKS, BLOCKS>>,
        >>();
        let adapter_width = Rv64VecHeapAdapterCols::<F, NUM_READS, BLOCKS, BLOCKS>::width();
        let width = adapter_width + BaseAir::<F>::width(&self.cpu.inner.expr);
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, layout);
        let cpu_ctx = Chip::<_, CpuBackend<SC>>::generate_proving_ctx(&self.cpu, matrix_arena);
        cpu_proving_ctx_to_gpu(cpu_ctx, &self.device_ctx)
    }
}

#[derive(Clone, Copy, Default)]
pub struct EccHybridProverExt;

/// Concrete inventory visitor for arena-free Weierstrass checkpoint tracegen.
///
/// Multiple configured curves may have the same Rust chip type, so coverage is
/// tracked by each chip's concrete opcode base rather than by downcast type.
#[cfg(feature = "rvr")]
pub struct WeierstrassPreflightGpuTracegen<'a> {
    program: &'a GpuPostflightProgram,
    transcript: &'a GpuPostflightTranscript,
    replay_plan: &'a GpuPostflightPlan,
    claimed_opcodes: Vec<u32>,
    pending_opcodes: BTreeSet<u32>,
}

#[cfg(feature = "rvr")]
impl<'a> WeierstrassPreflightGpuTracegen<'a> {
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
                PostflightAccessSpan::write_fixed_from_residuals(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    2,
                    blocks as u32,
                ),
            ];
            for local in [
                Rv64WeierstrassOpcode::EC_ADD_NE,
                Rv64WeierstrassOpcode::SETUP_EC_ADD_NE,
            ] {
                registry.register(
                    opcode(local)?,
                    &[2, 3, 1],
                    (1 << 6) | (1 << 7),
                    4,
                    5,
                    &add_spans,
                )?;
            }
            let double_spans = [
                PostflightAccessSpan::read_fixed(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    0,
                    blocks as u32,
                ),
                PostflightAccessSpan::write_fixed_from_residuals(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    1,
                    blocks as u32,
                ),
            ];
            registry.register(
                opcode(Rv64WeierstrassOpcode::EC_DOUBLE)?,
                &[2, 1],
                (1 << 3) | (1 << 6) | (1 << 7),
                4,
                5,
                &double_spans,
            )?;
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
                &[2, 1],
                (1 << 3) | (1 << 6) | (1 << 7),
                4,
                5,
                &setup_double_spans,
            )?;
        }
        Ok(())
    }

    /// Uploads one concrete RV64+Algebra+Weierstrass checkpoint program.
    pub fn upload_postflight_program<T: PrimeField32>(
        program: &Program<T>,
        memory_config: &MemoryConfig,
        modular: &ModularExtension,
        fp2: Option<&Fp2Extension>,
        weierstrass: &WeierstrassExtension,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<GpuPostflightProgram, GpuPostflightError> {
        let mut registry = PostflightAccessRegistry::default();
        AlgebraPreflightGpuTracegen::register_postflight_access_schedules(
            &mut registry,
            modular,
            fp2,
        )?;
        Self::register_postflight_access_schedules(&mut registry, weierstrass)?;
        registry
            .validate_no_native_collisions(Rv64ImPreflightGpuTracegen::postflight_opcode_bases())?;
        GpuPostflightProgram::upload_with_postflight_access_registry(
            program,
            memory_config,
            &registry,
            device_ctx,
        )
    }

    pub fn postflight<VB>(
        vm: &VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
        program: &GpuPostflightProgram,
        execution: &PreflightExecution,
        expected_retired: u32,
    ) -> Result<(GpuPostflightTranscript, GpuPostflightPlan), GpuPostflightError>
    where
        VB: VmBuilder<
            GpuBabyBearPoseidon2Engine,
            RecordArena = DenseRecordArena,
            SystemChipInventory = SystemChipInventoryGPU,
        >,
    {
        vm.postflight(
            program,
            execution,
            expected_retired,
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
                "Weierstrass inventory chip has no checkpoint replay configuration".to_string(),
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
    /// from one checkpoint transcript, without constructing execution records.
    pub fn generate_proving_ctx<VB>(
        mut self,
        vm: &mut VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
        modular: &ModularExtension,
        fp2: Option<&Fp2Extension>,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError>
    where
        VB: VmBuilder<
            GpuBabyBearPoseidon2Engine,
            RecordArena = DenseRecordArena,
            SystemChipInventory = SystemChipInventoryGPU,
        >,
    {
        let mut algebra = AlgebraPreflightGpuTracegen::new(
            self.program,
            self.transcript,
            self.replay_plan,
            modular,
            fp2,
        )
        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        let mut extension_opcodes = self.claimed_opcodes.clone();
        extension_opcodes.extend_from_slice(algebra.extension_opcodes());
        let mut rv64 = Rv64ImPreflightGpuTracegen::new_after_claiming_extension_opcodes(
            self.program,
            self.transcript,
            self.replay_plan,
            &extension_opcodes,
        )
        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        let ctx = vm.generate_preflight_proving_ctx_unchecked_coverage(
            self.program,
            self.transcript,
            self.replay_plan,
            |insertion_idx, chip| {
                if let Some(ctx) = self
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
                    rv64.generate_for_chip(insertion_idx, chip)
                        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
                }
            },
        )?;
        rv64.finish()
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        algebra
            .finish()
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        self.finish()
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        vm.complete_preflight_tracegen_session();
        Ok(ctx)
    }
}

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, WeierstrassExtension>
    for EccHybridProverExt
{
    fn extend_prover(
        &self,
        extension: &WeierstrassExtension,
        inventory: &mut ChipInventory<SC, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker_gpu = get_inventory_range_checker(inventory);
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let range_checker = range_checker_gpu.cpu_chip.clone().unwrap();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

        let device_ctx = range_checker_gpu.device_ctx.clone();

        for (curve_idx, curve) in extension.supported_curves.iter().enumerate() {
            let bytes = curve.modulus.bits().div_ceil(8) as usize;
            #[cfg(feature = "rvr")]
            let opcode_base =
                Rv64WeierstrassOpcode::CLASS_OFFSET + curve_idx * Rv64WeierstrassOpcode::COUNT;
            #[cfg(not(feature = "rvr"))]
            let _ = curve_idx;

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
                #[cfg(feature = "rvr")]
                let addne = HybridWeierstrassChip::new_with_replay(
                    addne,
                    device_ctx.clone(),
                    opcode_base,
                    range_checker_gpu.clone(),
                );
                #[cfg(not(feature = "rvr"))]
                let addne = HybridWeierstrassChip::new(addne, device_ctx.clone());
                inventory.add_executor_chip(addne);

                inventory.next_air::<WeierstrassAir<1, ECC_BLOCKS_32>>()?;
                let double = get_ec_double_chip::<F, ECC_BLOCKS_32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                    curve.a.clone(),
                );
                #[cfg(feature = "rvr")]
                let double = HybridWeierstrassChip::new_with_replay(
                    double,
                    device_ctx.clone(),
                    opcode_base,
                    range_checker_gpu.clone(),
                );
                #[cfg(not(feature = "rvr"))]
                let double = HybridWeierstrassChip::new(double, device_ctx.clone());
                inventory.add_executor_chip(double);
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
                #[cfg(feature = "rvr")]
                let addne = HybridWeierstrassChip::new_with_replay(
                    addne,
                    device_ctx.clone(),
                    opcode_base,
                    range_checker_gpu.clone(),
                );
                #[cfg(not(feature = "rvr"))]
                let addne = HybridWeierstrassChip::new(addne, device_ctx.clone());
                inventory.add_executor_chip(addne);

                inventory.next_air::<WeierstrassAir<1, ECC_BLOCKS_48>>()?;
                let double = get_ec_double_chip::<F, ECC_BLOCKS_48>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                    curve.a.clone(),
                );
                #[cfg(feature = "rvr")]
                let double = HybridWeierstrassChip::new_with_replay(
                    double,
                    device_ctx.clone(),
                    opcode_base,
                    range_checker_gpu.clone(),
                );
                #[cfg(not(feature = "rvr"))]
                let double = HybridWeierstrassChip::new(double, device_ctx.clone());
                inventory.add_executor_chip(double);
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

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Rv64WeierstrassHybridBuilder {
    type VmConfig = Rv64WeierstrassConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv64WeierstrassConfig,
        circuit: AirInventory<SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, GpuBackend, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &Rv64ModularHybridBuilder,
            &config.modular,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(
            &EccHybridProverExt,
            &config.weierstrass,
            inventory,
        )?;

        Ok(chip_complex)
    }
}
