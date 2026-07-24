//! Prover extension for the GPU backend which still does trace generation on CPU.

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
        cuda::vec_heap::{
            gather_vec_heap_trace_inputs, generate_field_expression_ctx_from_projection,
        },
        AlgebraRvrGpuTracegen, Fp2Extension, ModularExtension,
    },
    openvm_circuit::arch::{
        rvr::{
            cuda::{
                GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript,
                RvrCheckpointAccessRegistry, RvrCheckpointAccessSpan,
            },
            RvrCheckpointPreflightExecution,
        },
        GenerationError, VirtualMachine,
    },
    openvm_circuit_primitives::AnyChip,
    openvm_ecc_transpiler::Rv64WeierstrassOpcode,
    openvm_instructions::{program::Program, LocalOpcode},
    openvm_riscv_circuit::Rv64ImRvrGpuTracegen,
    openvm_stark_backend::{p3_field::PrimeField32, prover::ProvingContext},
    std::collections::BTreeSet,
    strum::EnumCount,
};

use crate::{
    get_ec_addne_chip, get_ec_double_chip, EccRecord, Rv64WeierstrassConfig, WeierstrassAir,
    WeierstrassChip, WeierstrassExtension, ECC_BLOCKS_32, ECC_BLOCKS_48, NUM_LIMBS_32,
    NUM_LIMBS_48,
};

pub struct HybridWeierstrassChip<F, const NUM_READS: usize, const BLOCKS: usize> {
    cpu: WeierstrassChip<F, NUM_READS, BLOCKS>,
    device_ctx: GpuDeviceCtx,
    #[cfg(feature = "rvr")]
    replay: Option<WeierstrassReplayConfig>,
}

#[cfg(feature = "rvr")]
#[derive(Clone, Copy)]
struct WeierstrassReplayConfig {
    opcode_base: usize,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
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
        pointer_max_bits: usize,
        timestamp_max_bits: usize,
    ) -> Self {
        Self {
            cpu,
            device_ctx,
            replay: Some(WeierstrassReplayConfig {
                opcode_base,
                pointer_max_bits,
                timestamp_max_bits,
            }),
        }
    }

    #[cfg(feature = "rvr")]
    fn local_opcodes() -> Result<[usize; 2], GpuRvrInputError> {
        match NUM_READS {
            2 => Ok([
                Rv64WeierstrassOpcode::EC_ADD_NE as usize,
                Rv64WeierstrassOpcode::SETUP_EC_ADD_NE as usize,
            ]),
            1 => Ok([
                Rv64WeierstrassOpcode::EC_DOUBLE as usize,
                Rv64WeierstrassOpcode::SETUP_EC_DOUBLE as usize,
            ]),
            _ => Err(GpuRvrInputError::InvalidTranscript(format!(
                "unsupported Weierstrass replay read count {NUM_READS}"
            ))),
        }
    }

    #[cfg(feature = "rvr")]
    pub fn opcode_base(&self) -> Option<usize> {
        self.replay.map(|replay| replay.opcode_base)
    }

    #[cfg(feature = "rvr")]
    pub fn generate_proving_ctx_from_rvr(
        &self,
        program: &GpuRvrProgram,
        transcript: &GpuRvrTranscript,
        replay_plan: &GpuRvrReplayPlan,
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        let replay = self.replay.ok_or_else(|| {
            GpuRvrInputError::InvalidTranscript(
                "Weierstrass chip was constructed without checkpoint replay".to_string(),
            )
        })?;
        let local_opcodes = Self::local_opcodes()?;
        let projection = gather_vec_heap_trace_inputs::<NUM_READS, BLOCKS>(
            program,
            transcript,
            replay_plan,
            replay.opcode_base,
            &local_opcodes,
            replay.pointer_max_bits,
            &self.device_ctx,
        )?;
        generate_field_expression_ctx_from_projection(
            &self.cpu,
            projection,
            replay.timestamp_max_bits,
            &self.device_ctx,
        )
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
pub struct WeierstrassRvrGpuTracegen<'a> {
    program: &'a GpuRvrProgram,
    transcript: &'a GpuRvrTranscript,
    replay_plan: &'a GpuRvrReplayPlan,
    claimed_opcodes: Vec<u32>,
    pending_opcodes: BTreeSet<u32>,
}

#[cfg(feature = "rvr")]
impl<'a> WeierstrassRvrGpuTracegen<'a> {
    #[doc(hidden)]
    pub fn register_checkpoint_access_schedules(
        registry: &mut RvrCheckpointAccessRegistry,
        extension: &WeierstrassExtension,
    ) -> Result<(), GpuRvrInputError> {
        for (curve_idx, curve) in extension.supported_curves.iter().enumerate() {
            let bytes = curve.modulus.bits().div_ceil(8) as usize;
            let blocks = if bytes <= NUM_LIMBS_32 {
                ECC_BLOCKS_32
            } else if bytes <= NUM_LIMBS_48 {
                ECC_BLOCKS_48
            } else {
                return Err(GpuRvrInputError::InvalidAccessSchedule(format!(
                    "Weierstrass curve {curve_idx} exceeds the supported 48-byte layout"
                )));
            };
            let opcode_base = Rv64WeierstrassOpcode::CLASS_OFFSET
                .checked_add(
                    curve_idx
                        .checked_mul(Rv64WeierstrassOpcode::COUNT)
                        .ok_or_else(|| {
                            GpuRvrInputError::InvalidAccessSchedule(
                                "Weierstrass opcode range overflow".to_string(),
                            )
                        })?,
                )
                .ok_or_else(|| {
                    GpuRvrInputError::InvalidAccessSchedule(
                        "Weierstrass opcode range overflow".to_string(),
                    )
                })?;
            let opcode = |local: Rv64WeierstrassOpcode| {
                let opcode = opcode_base.checked_add(local as usize).ok_or_else(|| {
                    GpuRvrInputError::InvalidAccessSchedule(
                        "Weierstrass opcode range overflow".to_string(),
                    )
                })?;
                u32::try_from(opcode).map_err(|_| GpuRvrInputError::OpcodeTooLarge(opcode))
            };
            let add_spans = [
                RvrCheckpointAccessSpan::read_fixed(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    0,
                    blocks as u32,
                ),
                RvrCheckpointAccessSpan::read_fixed(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    1,
                    blocks as u32,
                ),
                RvrCheckpointAccessSpan::write_fixed_from_residuals(
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
                RvrCheckpointAccessSpan::read_fixed(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    0,
                    blocks as u32,
                ),
                RvrCheckpointAccessSpan::write_fixed_from_residuals(
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

            // SETUP_EC_DOUBLE writes the configured field-expression program's
            // output without emitting residuals. The finite checkpoint schedule
            // ABI currently has no computed-write source, so it deliberately
            // remains unscheduled and fails closed during replay.
        }
        Ok(())
    }

    /// Uploads one concrete RV64+Algebra+Weierstrass checkpoint program.
    pub fn upload_checkpoint_program<T: PrimeField32>(
        program: &Program<T>,
        memory_config: &MemoryConfig,
        modular: &ModularExtension,
        fp2: Option<&Fp2Extension>,
        weierstrass: &WeierstrassExtension,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<GpuRvrProgram, GpuRvrInputError> {
        let mut registry = RvrCheckpointAccessRegistry::default();
        AlgebraRvrGpuTracegen::register_checkpoint_access_schedules(&mut registry, modular, fp2)?;
        Self::register_checkpoint_access_schedules(&mut registry, weierstrass)?;
        registry.validate_no_native_collisions(Rv64ImRvrGpuTracegen::checkpoint_opcode_bases())?;
        GpuRvrProgram::upload_with_checkpoint_access_registry(
            program,
            memory_config,
            &registry,
            device_ctx,
        )
    }

    pub fn expand_checkpoint_replay<VB>(
        vm: &VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
        program: &GpuRvrProgram,
        execution: &RvrCheckpointPreflightExecution,
        expected_retired: u32,
    ) -> Result<(GpuRvrTranscript, GpuRvrReplayPlan), GpuRvrInputError>
    where
        VB: VmBuilder<
            GpuBabyBearPoseidon2Engine,
            RecordArena = DenseRecordArena,
            SystemChipInventory = SystemChipInventoryGPU,
        >,
    {
        vm.expand_rvr_checkpoint_replay(
            program,
            execution,
            expected_retired,
            Rv64ImRvrGpuTracegen::checkpoint_opcode_bases(),
        )
    }

    pub fn new(
        extension: &WeierstrassExtension,
        program: &'a GpuRvrProgram,
        transcript: &'a GpuRvrTranscript,
        replay_plan: &'a GpuRvrReplayPlan,
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
    ) -> Result<AirProvingContext<GpuBackend>, GpuRvrInputError> {
        let base = chip.opcode_base().ok_or_else(|| {
            GpuRvrInputError::InvalidTranscript(
                "Weierstrass inventory chip has no checkpoint replay configuration".to_string(),
            )
        })?;
        for local in HybridWeierstrassChip::<F, NUM_READS, BLOCKS>::local_opcodes()? {
            let opcode = u32::try_from(base + local)
                .map_err(|_| GpuRvrInputError::OpcodeTooLarge(base + local))?;
            self.pending_opcodes.remove(&opcode);
        }
        chip.generate_proving_ctx_from_rvr(self.program, self.transcript, self.replay_plan)
    }

    /// Returns `Some` only for a Weierstrass-owned AIR, allowing a concrete
    /// combined coordinator to fall through to algebra and RV64 producers.
    pub fn generate_for_chip(
        &mut self,
        chip: &dyn AnyChip<DenseRecordArena, GpuBackend>,
    ) -> Result<Option<AirProvingContext<GpuBackend>>, GpuRvrInputError> {
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<HybridWeierstrassChip<F, 2, ECC_BLOCKS_32>>()
        {
            return self.generate_for_weierstrass_chip(chip).map(Some);
        }
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<HybridWeierstrassChip<F, 1, ECC_BLOCKS_32>>()
        {
            return self.generate_for_weierstrass_chip(chip).map(Some);
        }
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<HybridWeierstrassChip<F, 2, ECC_BLOCKS_48>>()
        {
            return self.generate_for_weierstrass_chip(chip).map(Some);
        }
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<HybridWeierstrassChip<F, 1, ECC_BLOCKS_48>>()
        {
            return self.generate_for_weierstrass_chip(chip).map(Some);
        }
        Ok(None)
    }

    pub fn finish(self) -> Result<(), GpuRvrInputError> {
        if self.pending_opcodes.is_empty() {
            Ok(())
        } else {
            Err(GpuRvrInputError::InvalidTranscript(format!(
                "Weierstrass RVR GPU tracegen did not visit opcodes {:?}",
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
        let mut algebra = AlgebraRvrGpuTracegen::new(
            self.program,
            self.transcript,
            self.replay_plan,
            modular,
            fp2,
        )
        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        let mut extension_opcodes = self.claimed_opcodes.clone();
        extension_opcodes.extend_from_slice(algebra.extension_opcodes());
        let mut rv64 = Rv64ImRvrGpuTracegen::new_after_claiming_extension_opcodes(
            self.program,
            self.transcript,
            self.replay_plan,
            &extension_opcodes,
        )
        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        let ctx = vm.generate_proving_ctx_from_rvr_unchecked_coverage(
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
        vm.complete_rvr_tracegen_session();
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
                    byte_ptr_max_bits,
                    timestamp_max_bits,
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
                    byte_ptr_max_bits,
                    timestamp_max_bits,
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
                    byte_ptr_max_bits,
                    timestamp_max_bits,
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
                    byte_ptr_max_bits,
                    timestamp_max_bits,
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

/// This builder will do tracegen for the RV64IM extensions on GPU but the modular and ecc
/// extensions on CPU.
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
