//! Prover extension for the GPU backend which still does trace generation on CPU.

use openvm_algebra_transpiler::{Fp2Opcode, Rv64ModularArithmeticOpcode};
use openvm_circuit::{
    arch::*,
    system::{
        cuda::{
            extensions::{get_inventory_range_checker, SystemGpuBuilder},
            SystemChipInventoryGPU,
        },
        memory::SharedMemoryHelper,
    },
};
use openvm_circuit_primitives::{
    bigint::utils::big_uint_to_limbs, hybrid_chip::cpu_proving_ctx_to_gpu, Chip,
};
use openvm_cpu_backend::CpuBackend;
use openvm_cuda_backend::{
    base::DeviceMatrix,
    prelude::{F, SC},
    BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend,
};
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_instructions::LocalOpcode;
use openvm_mod_circuit_builder::{ExprBuilderConfig, FieldExpressionMetadata};
use openvm_riscv_adapters::{
    Rv64IsEqualModU16AdapterCols, Rv64IsEqualModU16AdapterExecutor, Rv64IsEqualModU16AdapterFiller,
    Rv64IsEqualModU16AdapterRecord, Rv64VecHeapAdapterCols, Rv64VecHeapAdapterExecutor,
};
use openvm_riscv_circuit::{adapters::U16_BITS, Rv64ImGpuProverExt};
use openvm_stark_backend::{p3_air::BaseAir, prover::AirProvingContext};
use strum::EnumCount;
#[cfg(feature = "rvr")]
use {
    openvm_circuit::arch::rvr::cuda::{
        GpuRvrInputError, GpuRvrProgram, GpuRvrReplayPlan, GpuRvrTranscript,
        RvrCheckpointAccessRegistry, RvrCheckpointAccessSpan,
    },
    openvm_circuit::arch::rvr::RvrCheckpointPreflightExecution,
    openvm_circuit_primitives::AnyChip,
    openvm_instructions::program::Program,
    openvm_riscv_circuit::Rv64ImRvrGpuTracegen,
    openvm_stark_backend::{p3_field::PrimeField32, prover::ProvingContext},
    std::collections::BTreeSet,
};

use crate::{
    fp2_chip::{get_fp2_addsub_chip, get_fp2_muldiv_chip, Fp2Air, Fp2Chip},
    modular_chip::*,
    AlgebraRecord, Fp2Extension, ModularExtension, Rv64ModularConfig, Rv64ModularWithFp2Config,
    FP2_BLOCKS_32, FP2_BLOCKS_48, MODULAR_BLOCKS_32, MODULAR_BLOCKS_48, NUM_LIMBS_32,
    NUM_LIMBS_32_U16, NUM_LIMBS_48, NUM_LIMBS_48_U16,
};

pub struct HybridModularChip<F, const BLOCKS: usize> {
    cpu: ModularChip<F, BLOCKS>,
    device_ctx: GpuDeviceCtx,
    #[cfg(feature = "rvr")]
    replay: Option<FieldExpressionReplayConfig>,
    #[cfg(feature = "rvr")]
    direct_addsub: Option<crate::cuda::modular_addsub::ModularAddSubReplayChipGpu<BLOCKS>>,
}

#[cfg(feature = "rvr")]
#[derive(Clone, Copy)]
struct FieldExpressionReplayConfig {
    opcode_base: usize,
    pointer_max_bits: usize,
    timestamp_max_bits: usize,
}

#[cfg(feature = "rvr")]
fn validate_modular_is_eq_destinations<F: PrimeField32>(
    program: &Program<F>,
    num_moduli: usize,
) -> Result<(), GpuRvrInputError> {
    let opcode_base = Rv64ModularArithmeticOpcode::CLASS_OFFSET;
    let opcode_count = Rv64ModularArithmeticOpcode::COUNT;
    for (slot, entry) in program.instructions_and_debug_infos.iter().enumerate() {
        let Some((instruction, _)) = entry else {
            continue;
        };
        let opcode = instruction.opcode.as_usize();
        let Some(relative) = opcode.checked_sub(opcode_base) else {
            continue;
        };
        let local = relative % opcode_count;
        if relative / opcode_count < num_moduli
            && matches!(
                Rv64ModularArithmeticOpcode::from_usize(local),
                Rv64ModularArithmeticOpcode::IS_EQ | Rv64ModularArithmeticOpcode::SETUP_ISEQ
            )
            && instruction.a.as_canonical_u32() == 0
        {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "modular is-equal destination is x0 at program slot {slot}"
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "rvr")]
fn checked_replay_opcode(
    base: usize,
    local: usize,
) -> Result<openvm_instructions::VmOpcode, GpuRvrInputError> {
    let opcode = base.checked_add(local).ok_or_else(|| {
        GpuRvrInputError::InvalidTranscript("field-expression opcode overflow".to_string())
    })?;
    u32::try_from(opcode).map_err(|_| GpuRvrInputError::OpcodeTooLarge(opcode))?;
    Ok(openvm_instructions::VmOpcode::from_usize(opcode))
}

impl<const BLOCKS: usize> HybridModularChip<F, BLOCKS> {
    pub fn new(cpu: ModularChip<F, BLOCKS>, device_ctx: GpuDeviceCtx) -> Self {
        Self {
            cpu,
            device_ctx,
            #[cfg(feature = "rvr")]
            replay: None,
            #[cfg(feature = "rvr")]
            direct_addsub: None,
        }
    }

    #[cfg(feature = "rvr")]
    pub fn new_with_replay(
        cpu: ModularChip<F, BLOCKS>,
        device_ctx: GpuDeviceCtx,
        opcode_base: usize,
        pointer_max_bits: usize,
        timestamp_max_bits: usize,
    ) -> Self {
        Self {
            cpu,
            device_ctx,
            replay: Some(FieldExpressionReplayConfig {
                opcode_base,
                pointer_max_bits,
                timestamp_max_bits,
            }),
            direct_addsub: None,
        }
    }

    #[cfg(feature = "rvr")]
    #[allow(clippy::too_many_arguments)]
    pub fn new_addsub_with_replay(
        cpu: ModularChip<F, BLOCKS>,
        device_ctx: GpuDeviceCtx,
        modulus: &num_bigint::BigUint,
        opcode_base: usize,
        pointer_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker: std::sync::Arc<
            openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU,
        >,
    ) -> Self {
        let direct_addsub = crate::cuda::modular_addsub::ModularAddSubReplayChipGpu::new(
            &cpu,
            modulus,
            opcode_base,
            pointer_max_bits,
            timestamp_max_bits,
            range_checker,
        )
        .expect("valid modular add/sub replay configuration");
        Self {
            cpu,
            device_ctx,
            replay: Some(FieldExpressionReplayConfig {
                opcode_base,
                pointer_max_bits,
                timestamp_max_bits,
            }),
            direct_addsub,
        }
    }

    #[cfg(feature = "rvr")]
    pub fn generate_proving_ctx_from_rvr(
        &self,
        program: &openvm_circuit::arch::rvr::cuda::GpuRvrProgram,
        transcript: &openvm_circuit::arch::rvr::cuda::GpuRvrTranscript,
        replay_plan: &openvm_circuit::arch::rvr::cuda::GpuRvrReplayPlan,
    ) -> Result<AirProvingContext<GpuBackend>, openvm_circuit::arch::rvr::cuda::GpuRvrInputError>
    {
        if let Some(direct) = &self.direct_addsub {
            return direct.generate_proving_ctx(program, transcript, replay_plan);
        }
        let replay = self.replay.ok_or_else(|| {
            openvm_circuit::arch::rvr::cuda::GpuRvrInputError::InvalidTranscript(
                "Modular chip was constructed without checkpoint replay".to_string(),
            )
        })?;
        let projection = crate::cuda::vec_heap::gather_vec_heap_trace_inputs::<2, BLOCKS>(
            program,
            transcript,
            replay_plan,
            replay.opcode_base,
            &self.cpu.inner.local_opcode_idx,
            replay.pointer_max_bits,
            &self.device_ctx,
        )?;
        crate::cuda::vec_heap::generate_field_expression_ctx_from_projection(
            &self.cpu,
            projection,
            replay.timestamp_max_bits,
            &self.device_ctx,
        )
    }

    #[cfg(all(test, feature = "rvr"))]
    pub(crate) fn uses_direct_addsub_replay(&self) -> bool {
        self.direct_addsub.is_some()
    }

    #[cfg(feature = "rvr")]
    fn checkpoint_opcodes(
        &self,
    ) -> Result<Vec<openvm_instructions::VmOpcode>, openvm_circuit::arch::rvr::cuda::GpuRvrInputError>
    {
        let replay = self.replay.ok_or_else(|| {
            openvm_circuit::arch::rvr::cuda::GpuRvrInputError::InvalidTranscript(
                "Modular chip was constructed without checkpoint replay".to_string(),
            )
        })?;
        Ok(self
            .cpu
            .inner
            .local_opcode_idx
            .iter()
            .map(|&local| checked_replay_opcode(replay.opcode_base, local))
            .collect::<Result<Vec<_>, _>>()?)
    }
}

// Auto-implementation of Chip for GpuBackend for a Cpu Chip by doing conversion
// of Dense->Matrix Record Arena, cpu tracegen, and then H2D transfer of the trace matrix.
impl<const BLOCKS: usize> Chip<DenseRecordArena, GpuBackend> for HybridModularChip<F, BLOCKS> {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.program().canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv64VecHeapAdapterExecutor<2, BLOCKS, BLOCKS>,
        >::new(total_input_limbs));

        let record_size =
            RecordSeeker::<DenseRecordArena, AlgebraRecord<2, BLOCKS>, _>::get_aligned_record_size(
                &layout,
            );

        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;

        let height = num_records.next_power_of_two();
        let mut seeker = arena.get_record_seeker::<AlgebraRecord<2, BLOCKS>, AdapterCoreLayout<
            FieldExpressionMetadata<F, Rv64VecHeapAdapterExecutor<2, BLOCKS, BLOCKS>>,
        >>();
        let adapter_width = Rv64VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS>::width();
        let width = adapter_width + BaseAir::<F>::width(&self.cpu.inner.expr);
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, layout);
        let cpu_ctx = Chip::<_, CpuBackend<SC>>::generate_proving_ctx(&self.cpu, matrix_arena);
        cpu_proving_ctx_to_gpu(cpu_ctx, &self.device_ctx)
    }
}

pub struct HybridModularIsEqualChip<F, const NUM_LANES: usize, const TOTAL_LIMBS: usize> {
    cpu: ModularIsEqualU16Chip<F, NUM_LANES, TOTAL_LIMBS>,
    device_ctx: GpuDeviceCtx,
    #[cfg(feature = "rvr")]
    replay: Option<crate::cuda::ModularIsEqualReplayChipGpu<NUM_LANES, TOTAL_LIMBS>>,
}

impl<const NUM_LANES: usize, const TOTAL_LIMBS: usize>
    HybridModularIsEqualChip<F, NUM_LANES, TOTAL_LIMBS>
{
    pub fn new(
        cpu: ModularIsEqualU16Chip<F, NUM_LANES, TOTAL_LIMBS>,
        device_ctx: GpuDeviceCtx,
    ) -> Self {
        Self {
            cpu,
            device_ctx,
            #[cfg(feature = "rvr")]
            replay: None,
        }
    }

    #[cfg(feature = "rvr")]
    pub fn new_with_replay(
        cpu: ModularIsEqualU16Chip<F, NUM_LANES, TOTAL_LIMBS>,
        device_ctx: GpuDeviceCtx,
        modulus_limbs: [u16; TOTAL_LIMBS],
        opcode_base: usize,
        pointer_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker_gpu: std::sync::Arc<
            openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU,
        >,
    ) -> Self {
        Self {
            cpu,
            device_ctx,
            replay: Some(crate::cuda::ModularIsEqualReplayChipGpu::new(
                modulus_limbs,
                opcode_base,
                pointer_max_bits,
                timestamp_max_bits,
                range_checker_gpu,
            )),
        }
    }

    #[cfg(feature = "rvr")]
    pub fn generate_proving_ctx_from_rvr(
        &self,
        program: &openvm_circuit::arch::rvr::cuda::GpuRvrProgram,
        transcript: &openvm_circuit::arch::rvr::cuda::GpuRvrTranscript,
        replay_plan: &openvm_circuit::arch::rvr::cuda::GpuRvrReplayPlan,
    ) -> Result<AirProvingContext<GpuBackend>, openvm_circuit::arch::rvr::cuda::GpuRvrInputError>
    {
        self.replay
            .as_ref()
            .ok_or_else(|| {
                openvm_circuit::arch::rvr::cuda::GpuRvrInputError::InvalidTranscript(
                    "ModularIsEqual chip was constructed without checkpoint replay".to_string(),
                )
            })?
            .generate_proving_ctx_from_rvr(program, transcript, replay_plan)
    }

    #[cfg(feature = "rvr")]
    fn checkpoint_opcodes(
        &self,
    ) -> Result<[openvm_instructions::VmOpcode; 2], openvm_circuit::arch::rvr::cuda::GpuRvrInputError>
    {
        self.replay
            .as_ref()
            .ok_or_else(|| {
                openvm_circuit::arch::rvr::cuda::GpuRvrInputError::InvalidTranscript(
                    "ModularIsEqual chip was constructed without checkpoint replay".to_string(),
                )
            })?
            .checkpoint_opcodes()
    }
}

impl<const NUM_LANES: usize, const TOTAL_LIMBS: usize> Chip<DenseRecordArena, GpuBackend>
    for HybridModularIsEqualChip<F, NUM_LANES, TOTAL_LIMBS>
{
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let record_size = size_of::<(
            Rv64IsEqualModU16AdapterRecord<2, NUM_LANES>,
            ModularIsEqualRecord<TOTAL_LIMBS>,
        )>();
        let trace_width = Rv64IsEqualModU16AdapterCols::<F, 2, NUM_LANES>::width()
            + ModularIsEqualCoreCols::<F, TOTAL_LIMBS>::width();
        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let height = num_records.next_power_of_two();
        let mut seeker = arena.get_record_seeker::<(
            &mut Rv64IsEqualModU16AdapterRecord<2, NUM_LANES>,
            &mut ModularIsEqualRecord<TOTAL_LIMBS>,
        ), EmptyAdapterCoreLayout<
            F,
            Rv64IsEqualModU16AdapterExecutor<2, NUM_LANES, TOTAL_LIMBS>,
        >>();
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, trace_width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, EmptyAdapterCoreLayout::new());
        let cpu_ctx = Chip::<_, CpuBackend<SC>>::generate_proving_ctx(&self.cpu, matrix_arena);
        cpu_proving_ctx_to_gpu(cpu_ctx, &self.device_ctx)
    }
}

#[derive(Clone, Copy, Default)]
pub struct AlgebraHybridProverExt;

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, ModularExtension>
    for AlgebraHybridProverExt
{
    fn extend_prover(
        &self,
        extension: &ModularExtension,
        inventory: &mut ChipInventory<SC, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker_gpu = get_inventory_range_checker(inventory);
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let range_checker = range_checker_gpu.cpu_chip.clone().unwrap();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let device_ctx = range_checker_gpu.device_ctx.clone();

        for (i, modulus) in extension.supported_moduli.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8) as usize;
            let start_offset =
                Rv64ModularArithmeticOpcode::CLASS_OFFSET + i * Rv64ModularArithmeticOpcode::COUNT;

            let modulus_limbs = big_uint_to_limbs(modulus, U16_BITS);

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_32>>()?;
                let addsub = get_modular_addsub_chip::<F, MODULAR_BLOCKS_32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                #[cfg(feature = "rvr")]
                let addsub = HybridModularChip::new_addsub_with_replay(
                    addsub,
                    device_ctx.clone(),
                    modulus,
                    start_offset,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                );
                #[cfg(not(feature = "rvr"))]
                let addsub = HybridModularChip::new(addsub, device_ctx.clone());
                inventory.add_executor_chip(addsub);

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_32>>()?;
                let muldiv = get_modular_muldiv_chip::<F, MODULAR_BLOCKS_32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                #[cfg(feature = "rvr")]
                let muldiv = HybridModularChip::new_with_replay(
                    muldiv,
                    device_ctx.clone(),
                    start_offset,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                );
                #[cfg(not(feature = "rvr"))]
                let muldiv = HybridModularChip::new(muldiv, device_ctx.clone());
                inventory.add_executor_chip(muldiv);

                let modulus_limbs = std::array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u16
                    } else {
                        0
                    }
                });
                inventory
                    .next_air::<ModularIsEqualU16Air<MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>>()?;
                let is_eq = ModularIsEqualU16Chip::<F, MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>::new(
                    ModularIsEqualFiller::new(
                        Rv64IsEqualModU16AdapterFiller::new(
                            byte_ptr_max_bits,
                            range_checker.clone(),
                        ),
                        start_offset,
                        modulus_limbs,
                        range_checker.clone(),
                    ),
                    mem_helper.clone(),
                );
                #[cfg(feature = "rvr")]
                let is_eq = HybridModularIsEqualChip::new_with_replay(
                    is_eq,
                    device_ctx.clone(),
                    modulus_limbs,
                    start_offset,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                );
                #[cfg(not(feature = "rvr"))]
                let is_eq = HybridModularIsEqualChip::new(is_eq, device_ctx.clone());
                inventory.add_executor_chip(is_eq);
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_48>>()?;
                let addsub = get_modular_addsub_chip::<F, MODULAR_BLOCKS_48>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                #[cfg(feature = "rvr")]
                let addsub = HybridModularChip::new_addsub_with_replay(
                    addsub,
                    device_ctx.clone(),
                    modulus,
                    start_offset,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                );
                #[cfg(not(feature = "rvr"))]
                let addsub = HybridModularChip::new(addsub, device_ctx.clone());
                inventory.add_executor_chip(addsub);

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_48>>()?;
                let muldiv = get_modular_muldiv_chip::<F, MODULAR_BLOCKS_48>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                #[cfg(feature = "rvr")]
                let muldiv = HybridModularChip::new_with_replay(
                    muldiv,
                    device_ctx.clone(),
                    start_offset,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                );
                #[cfg(not(feature = "rvr"))]
                let muldiv = HybridModularChip::new(muldiv, device_ctx.clone());
                inventory.add_executor_chip(muldiv);

                let modulus_limbs = std::array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u16
                    } else {
                        0
                    }
                });
                inventory
                    .next_air::<ModularIsEqualU16Air<MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>>()?;
                let is_eq = ModularIsEqualU16Chip::<F, MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>::new(
                    ModularIsEqualFiller::new(
                        Rv64IsEqualModU16AdapterFiller::new(
                            byte_ptr_max_bits,
                            range_checker.clone(),
                        ),
                        start_offset,
                        modulus_limbs,
                        range_checker.clone(),
                    ),
                    mem_helper.clone(),
                );
                #[cfg(feature = "rvr")]
                let is_eq = HybridModularIsEqualChip::new_with_replay(
                    is_eq,
                    device_ctx.clone(),
                    modulus_limbs,
                    start_offset,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                );
                #[cfg(not(feature = "rvr"))]
                let is_eq = HybridModularIsEqualChip::new(is_eq, device_ctx.clone());
                inventory.add_executor_chip(is_eq);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

pub struct HybridFp2Chip<F, const BLOCKS: usize> {
    cpu: Fp2Chip<F, BLOCKS>,
    device_ctx: GpuDeviceCtx,
    #[cfg(feature = "rvr")]
    replay: Option<FieldExpressionReplayConfig>,
}

impl<const BLOCKS: usize> HybridFp2Chip<F, BLOCKS> {
    pub fn new(cpu: Fp2Chip<F, BLOCKS>, device_ctx: GpuDeviceCtx) -> Self {
        Self {
            cpu,
            device_ctx,
            #[cfg(feature = "rvr")]
            replay: None,
        }
    }

    #[cfg(feature = "rvr")]
    pub fn new_with_replay(
        cpu: Fp2Chip<F, BLOCKS>,
        device_ctx: GpuDeviceCtx,
        opcode_base: usize,
        pointer_max_bits: usize,
        timestamp_max_bits: usize,
    ) -> Self {
        Self {
            cpu,
            device_ctx,
            replay: Some(FieldExpressionReplayConfig {
                opcode_base,
                pointer_max_bits,
                timestamp_max_bits,
            }),
        }
    }

    #[cfg(feature = "rvr")]
    pub fn generate_proving_ctx_from_rvr(
        &self,
        program: &openvm_circuit::arch::rvr::cuda::GpuRvrProgram,
        transcript: &openvm_circuit::arch::rvr::cuda::GpuRvrTranscript,
        replay_plan: &openvm_circuit::arch::rvr::cuda::GpuRvrReplayPlan,
    ) -> Result<AirProvingContext<GpuBackend>, openvm_circuit::arch::rvr::cuda::GpuRvrInputError>
    {
        let replay = self.replay.ok_or_else(|| {
            openvm_circuit::arch::rvr::cuda::GpuRvrInputError::InvalidTranscript(
                "Fp2 chip was constructed without checkpoint replay".to_string(),
            )
        })?;
        let projection = crate::cuda::vec_heap::gather_vec_heap_trace_inputs::<2, BLOCKS>(
            program,
            transcript,
            replay_plan,
            replay.opcode_base,
            &self.cpu.inner.local_opcode_idx,
            replay.pointer_max_bits,
            &self.device_ctx,
        )?;
        crate::cuda::vec_heap::generate_field_expression_ctx_from_projection(
            &self.cpu,
            projection,
            replay.timestamp_max_bits,
            &self.device_ctx,
        )
    }

    #[cfg(feature = "rvr")]
    fn checkpoint_opcodes(
        &self,
    ) -> Result<Vec<openvm_instructions::VmOpcode>, openvm_circuit::arch::rvr::cuda::GpuRvrInputError>
    {
        let replay = self.replay.ok_or_else(|| {
            openvm_circuit::arch::rvr::cuda::GpuRvrInputError::InvalidTranscript(
                "Fp2 chip was constructed without checkpoint replay".to_string(),
            )
        })?;
        Ok(self
            .cpu
            .inner
            .local_opcode_idx
            .iter()
            .map(|&local| checked_replay_opcode(replay.opcode_base, local))
            .collect::<Result<Vec<_>, _>>()?)
    }
}

impl<const BLOCKS: usize> Chip<DenseRecordArena, GpuBackend> for HybridFp2Chip<F, BLOCKS> {
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.program().canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv64VecHeapAdapterExecutor<2, BLOCKS, BLOCKS>,
        >::new(total_input_limbs));

        let record_size =
            RecordSeeker::<DenseRecordArena, AlgebraRecord<2, BLOCKS>, _>::get_aligned_record_size(
                &layout,
            );

        let records = arena.allocated();
        if records.is_empty() {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let height = num_records.next_power_of_two();
        let mut seeker = arena.get_record_seeker::<AlgebraRecord<2, BLOCKS>, AdapterCoreLayout<
            FieldExpressionMetadata<F, Rv64VecHeapAdapterExecutor<2, BLOCKS, BLOCKS>>,
        >>();
        let adapter_width = Rv64VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS>::width();
        let width = adapter_width + BaseAir::<F>::width(&self.cpu.inner.expr);
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, layout);
        let cpu_ctx = Chip::<_, CpuBackend<SC>>::generate_proving_ctx(&self.cpu, matrix_arena);
        cpu_proving_ctx_to_gpu(cpu_ctx, &self.device_ctx)
    }
}

/// Concrete algebra checkpoint producers for the existing reverse inventory
/// walk. Coverage is derived from configured opcode ranges and fails closed.
#[cfg(feature = "rvr")]
pub struct AlgebraRvrGpuTracegen<'a> {
    program: &'a GpuRvrProgram,
    transcript: &'a GpuRvrTranscript,
    replay_plan: &'a GpuRvrReplayPlan,
    configured_opcodes: Vec<u32>,
    unclaimed: BTreeSet<u32>,
}

#[cfg(feature = "rvr")]
impl<'a> AlgebraRvrGpuTracegen<'a> {
    #[doc(hidden)]
    pub fn register_checkpoint_access_schedules(
        registry: &mut RvrCheckpointAccessRegistry,
        modular: &ModularExtension,
        fp2: Option<&Fp2Extension>,
    ) -> Result<(), GpuRvrInputError> {
        for (index, modulus) in modular.supported_moduli.iter().enumerate() {
            let bytes = modulus.bits().div_ceil(8) as usize;
            let blocks = if bytes <= NUM_LIMBS_32 {
                MODULAR_BLOCKS_32
            } else if bytes <= NUM_LIMBS_48 {
                MODULAR_BLOCKS_48
            } else {
                return Err(GpuRvrInputError::InvalidAccessSchedule(format!(
                    "modulus {index} exceeds the supported 48-byte layout"
                )));
            };
            let opcode_base = Rv64ModularArithmeticOpcode::CLASS_OFFSET
                .checked_add(
                    index
                        .checked_mul(Rv64ModularArithmeticOpcode::COUNT)
                        .ok_or_else(|| {
                            GpuRvrInputError::InvalidAccessSchedule(
                                "Modular opcode range overflow".to_string(),
                            )
                        })?,
                )
                .ok_or_else(|| {
                    GpuRvrInputError::InvalidAccessSchedule(
                        "Modular opcode range overflow".to_string(),
                    )
                })?;
            let opcode = |local: Rv64ModularArithmeticOpcode| {
                let opcode = opcode_base.checked_add(local as usize).ok_or_else(|| {
                    GpuRvrInputError::InvalidAccessSchedule(
                        "Modular opcode range overflow".to_string(),
                    )
                })?;
                u32::try_from(opcode).map_err(|_| GpuRvrInputError::OpcodeTooLarge(opcode))
            };
            let read_spans = [
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
            ];
            let write_spans = [
                read_spans[0],
                read_spans[1],
                RvrCheckpointAccessSpan::write_fixed_from_residuals(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    2,
                    blocks as u32,
                ),
            ];
            let zero_write_spans = [
                read_spans[0],
                read_spans[1],
                RvrCheckpointAccessSpan::write_fixed_zero(
                    openvm_instructions::riscv::RV64_MEMORY_AS,
                    2,
                    blocks as u32,
                ),
            ];
            for local in [
                Rv64ModularArithmeticOpcode::ADD,
                Rv64ModularArithmeticOpcode::SUB,
                Rv64ModularArithmeticOpcode::MUL,
                Rv64ModularArithmeticOpcode::DIV,
            ] {
                registry.register(
                    opcode(local)?,
                    &[2, 3, 1],
                    (1 << 6) | (1 << 7),
                    4,
                    5,
                    &write_spans,
                )?;
            }
            for local in [
                Rv64ModularArithmeticOpcode::SETUP_ADDSUB,
                Rv64ModularArithmeticOpcode::SETUP_MULDIV,
            ] {
                registry.register(
                    opcode(local)?,
                    &[2, 3, 1],
                    (1 << 6) | (1 << 7),
                    4,
                    5,
                    &zero_write_spans,
                )?;
            }
            registry.register_with_residual_register_write(
                opcode(Rv64ModularArithmeticOpcode::IS_EQ)?,
                &[2, 3],
                (1 << 6) | (1 << 7),
                4,
                5,
                &read_spans,
                1,
            )?;
            registry.register_with_zero_register_write(
                opcode(Rv64ModularArithmeticOpcode::SETUP_ISEQ)?,
                &[2, 3],
                (1 << 6) | (1 << 7),
                4,
                5,
                &read_spans,
                1,
            )?;
        }
        if let Some(fp2) = fp2 {
            for (index, (_, modulus)) in fp2.supported_moduli.iter().enumerate() {
                let bytes = modulus.bits().div_ceil(8) as usize;
                let blocks = if bytes <= NUM_LIMBS_32 {
                    FP2_BLOCKS_32
                } else if bytes <= NUM_LIMBS_48 {
                    FP2_BLOCKS_48
                } else {
                    return Err(GpuRvrInputError::InvalidAccessSchedule(format!(
                        "Fp2 modulus {index} exceeds the supported 48-byte layout"
                    )));
                };
                let opcode_base = Fp2Opcode::CLASS_OFFSET
                    .checked_add(index.checked_mul(Fp2Opcode::COUNT).ok_or_else(|| {
                        GpuRvrInputError::InvalidAccessSchedule(
                            "Fp2 opcode range overflow".to_string(),
                        )
                    })?)
                    .ok_or_else(|| {
                        GpuRvrInputError::InvalidAccessSchedule(
                            "Fp2 opcode range overflow".to_string(),
                        )
                    })?;
                let opcode = |local: Fp2Opcode| {
                    let opcode = opcode_base.checked_add(local as usize).ok_or_else(|| {
                        GpuRvrInputError::InvalidAccessSchedule(
                            "Fp2 opcode range overflow".to_string(),
                        )
                    })?;
                    u32::try_from(opcode).map_err(|_| GpuRvrInputError::OpcodeTooLarge(opcode))
                };
                let spans = [
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
                    Fp2Opcode::ADD,
                    Fp2Opcode::SUB,
                    Fp2Opcode::SETUP_ADDSUB,
                    Fp2Opcode::MUL,
                    Fp2Opcode::DIV,
                    Fp2Opcode::SETUP_MULDIV,
                ] {
                    registry.register(
                        opcode(local)?,
                        &[2, 3, 1],
                        (1 << 6) | (1 << 7),
                        4,
                        5,
                        &spans,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Uploads one concrete RV64+Algebra checkpoint program. The registry is
    /// immutable program metadata; execution still writes only checkpoints and
    /// irreducible postimages.
    pub fn upload_checkpoint_program<T: PrimeField32>(
        program: &Program<T>,
        memory_config: &MemoryConfig,
        modular: &ModularExtension,
        fp2: Option<&Fp2Extension>,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<GpuRvrProgram, GpuRvrInputError> {
        validate_modular_is_eq_destinations(program, modular.supported_moduli.len())?;
        let mut registry = RvrCheckpointAccessRegistry::default();
        Self::register_checkpoint_access_schedules(&mut registry, modular, fp2)?;
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
        program: &'a GpuRvrProgram,
        transcript: &'a GpuRvrTranscript,
        replay_plan: &'a GpuRvrReplayPlan,
        modular: &ModularExtension,
        fp2: Option<&Fp2Extension>,
    ) -> Result<Self, GpuRvrInputError> {
        let mut configured_opcodes = Vec::new();
        let mut configured = BTreeSet::new();
        for index in 0..modular.supported_moduli.len() {
            let stride = index
                .checked_mul(Rv64ModularArithmeticOpcode::COUNT)
                .ok_or_else(|| {
                    GpuRvrInputError::InvalidTranscript("Modular opcode range overflow".to_string())
                })?;
            let base = Rv64ModularArithmeticOpcode::CLASS_OFFSET
                .checked_add(stride)
                .ok_or_else(|| {
                    GpuRvrInputError::InvalidTranscript("Modular opcode range overflow".to_string())
                })?;
            for local in 0..Rv64ModularArithmeticOpcode::COUNT {
                let opcode = base.checked_add(local).ok_or_else(|| {
                    GpuRvrInputError::InvalidTranscript("Modular opcode overflow".to_string())
                })?;
                let opcode =
                    u32::try_from(opcode).map_err(|_| GpuRvrInputError::OpcodeTooLarge(opcode))?;
                if !configured.insert(opcode) {
                    return Err(GpuRvrInputError::InvalidTranscript(format!(
                        "duplicate Algebra opcode ownership for {opcode}"
                    )));
                }
                configured_opcodes.push(opcode);
            }
        }
        if let Some(fp2) = fp2 {
            for index in 0..fp2.supported_moduli.len() {
                let stride = index.checked_mul(Fp2Opcode::COUNT).ok_or_else(|| {
                    GpuRvrInputError::InvalidTranscript("Fp2 opcode range overflow".to_string())
                })?;
                let base = Fp2Opcode::CLASS_OFFSET.checked_add(stride).ok_or_else(|| {
                    GpuRvrInputError::InvalidTranscript("Fp2 opcode range overflow".to_string())
                })?;
                for local in 0..Fp2Opcode::COUNT {
                    let opcode = base.checked_add(local).ok_or_else(|| {
                        GpuRvrInputError::InvalidTranscript("Fp2 opcode overflow".to_string())
                    })?;
                    let opcode = u32::try_from(opcode)
                        .map_err(|_| GpuRvrInputError::OpcodeTooLarge(opcode))?;
                    if !configured.insert(opcode) {
                        return Err(GpuRvrInputError::InvalidTranscript(format!(
                            "duplicate Algebra opcode ownership for {opcode}"
                        )));
                    }
                    configured_opcodes.push(opcode);
                }
            }
        }
        let unclaimed = replay_plan
            .executed_opcodes()
            .filter(|opcode| configured.contains(opcode))
            .collect();
        Ok(Self {
            program,
            transcript,
            replay_plan,
            configured_opcodes,
            unclaimed,
        })
    }

    pub fn extension_opcodes(&self) -> &[u32] {
        &self.configured_opcodes
    }

    fn claim(&mut self, opcodes: impl IntoIterator<Item = openvm_instructions::VmOpcode>) {
        for opcode in opcodes {
            self.unclaimed.remove(&(opcode.as_usize() as u32));
        }
    }

    pub fn generate_for_chip(
        &mut self,
        chip: &dyn AnyChip<DenseRecordArena, GpuBackend>,
    ) -> Result<Option<AirProvingContext<GpuBackend>>, GpuRvrInputError> {
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<HybridModularChip<F, MODULAR_BLOCKS_32>>()
        {
            let opcodes = chip.checkpoint_opcodes()?;
            let ctx = chip.generate_proving_ctx_from_rvr(
                self.program,
                self.transcript,
                self.replay_plan,
            )?;
            self.claim(opcodes);
            return Ok(Some(ctx));
        }
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<HybridModularChip<F, MODULAR_BLOCKS_48>>()
        {
            let opcodes = chip.checkpoint_opcodes()?;
            let ctx = chip.generate_proving_ctx_from_rvr(
                self.program,
                self.transcript,
                self.replay_plan,
            )?;
            self.claim(opcodes);
            return Ok(Some(ctx));
        }
        if let Some(chip) =
            chip.as_any()
                .downcast_ref::<HybridModularIsEqualChip<F, MODULAR_BLOCKS_32, NUM_LIMBS_32_U16>>()
        {
            let opcodes = chip.checkpoint_opcodes()?;
            let ctx = chip.generate_proving_ctx_from_rvr(
                self.program,
                self.transcript,
                self.replay_plan,
            )?;
            self.claim(opcodes);
            return Ok(Some(ctx));
        }
        if let Some(chip) =
            chip.as_any()
                .downcast_ref::<HybridModularIsEqualChip<F, MODULAR_BLOCKS_48, NUM_LIMBS_48_U16>>()
        {
            let opcodes = chip.checkpoint_opcodes()?;
            let ctx = chip.generate_proving_ctx_from_rvr(
                self.program,
                self.transcript,
                self.replay_plan,
            )?;
            self.claim(opcodes);
            return Ok(Some(ctx));
        }
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<HybridFp2Chip<F, FP2_BLOCKS_32>>()
        {
            let opcodes = chip.checkpoint_opcodes()?;
            let ctx = chip.generate_proving_ctx_from_rvr(
                self.program,
                self.transcript,
                self.replay_plan,
            )?;
            self.claim(opcodes);
            return Ok(Some(ctx));
        }
        if let Some(chip) = chip
            .as_any()
            .downcast_ref::<HybridFp2Chip<F, FP2_BLOCKS_48>>()
        {
            let opcodes = chip.checkpoint_opcodes()?;
            let ctx = chip.generate_proving_ctx_from_rvr(
                self.program,
                self.transcript,
                self.replay_plan,
            )?;
            self.claim(opcodes);
            return Ok(Some(ctx));
        }
        Ok(None)
    }

    pub fn finish(self) -> Result<(), GpuRvrInputError> {
        if !self.unclaimed.is_empty() {
            return Err(GpuRvrInputError::InvalidTranscript(format!(
                "Algebra RVR GPU tracegen did not visit opcodes {:?}",
                self.unclaimed
            )));
        }
        Ok(())
    }

    pub fn generate_proving_ctx<VB>(
        mut self,
        vm: &mut VirtualMachine<GpuBabyBearPoseidon2Engine, VB>,
    ) -> Result<ProvingContext<GpuBackend>, GenerationError>
    where
        VB: VmBuilder<
            GpuBabyBearPoseidon2Engine,
            RecordArena = DenseRecordArena,
            SystemChipInventory = SystemChipInventoryGPU,
        >,
    {
        let extension_opcodes = self.configured_opcodes.clone();
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
                } else {
                    rv64.generate_for_chip(insertion_idx, chip)
                        .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))
                }
            },
        )?;
        rv64.finish()
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        self.finish()
            .map_err(|error| GenerationError::ExtensionTracegen(error.to_string()))?;
        vm.complete_rvr_tracegen_session();
        Ok(ctx)
    }
}

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Fp2Extension>
    for AlgebraHybridProverExt
{
    fn extend_prover(
        &self,
        extension: &Fp2Extension,
        inventory: &mut ChipInventory<SC, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker_gpu = get_inventory_range_checker(inventory);
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let byte_ptr_max_bits = to_byte_ptr_bits(inventory.airs().pointer_max_bits());
        let range_checker = range_checker_gpu.cpu_chip.clone().unwrap();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let device_ctx = range_checker_gpu.device_ctx.clone();

        for (i, (_, modulus)) in extension.supported_moduli.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8) as usize;
            let start_offset = Fp2Opcode::CLASS_OFFSET + i * Fp2Opcode::COUNT;

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };

                inventory.next_air::<Fp2Air<FP2_BLOCKS_32>>()?;
                let addsub = get_fp2_addsub_chip::<F, FP2_BLOCKS_32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                #[cfg(feature = "rvr")]
                let addsub = HybridFp2Chip::new_with_replay(
                    addsub,
                    device_ctx.clone(),
                    start_offset,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                );
                #[cfg(not(feature = "rvr"))]
                let addsub = HybridFp2Chip::new(addsub, device_ctx.clone());
                inventory.add_executor_chip(addsub);

                inventory.next_air::<Fp2Air<FP2_BLOCKS_32>>()?;
                let muldiv = get_fp2_muldiv_chip::<F, FP2_BLOCKS_32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                #[cfg(feature = "rvr")]
                let muldiv = HybridFp2Chip::new_with_replay(
                    muldiv,
                    device_ctx.clone(),
                    start_offset,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                );
                #[cfg(not(feature = "rvr"))]
                let muldiv = HybridFp2Chip::new(muldiv, device_ctx.clone());
                inventory.add_executor_chip(muldiv);
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };

                inventory.next_air::<Fp2Air<FP2_BLOCKS_48>>()?;
                let addsub = get_fp2_addsub_chip::<F, FP2_BLOCKS_48>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                #[cfg(feature = "rvr")]
                let addsub = HybridFp2Chip::new_with_replay(
                    addsub,
                    device_ctx.clone(),
                    start_offset,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                );
                #[cfg(not(feature = "rvr"))]
                let addsub = HybridFp2Chip::new(addsub, device_ctx.clone());
                inventory.add_executor_chip(addsub);

                inventory.next_air::<Fp2Air<FP2_BLOCKS_48>>()?;
                let muldiv = get_fp2_muldiv_chip::<F, FP2_BLOCKS_48>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    byte_ptr_max_bits,
                );
                #[cfg(feature = "rvr")]
                let muldiv = HybridFp2Chip::new_with_replay(
                    muldiv,
                    device_ctx.clone(),
                    start_offset,
                    byte_ptr_max_bits,
                    timestamp_max_bits,
                );
                #[cfg(not(feature = "rvr"))]
                let muldiv = HybridFp2Chip::new(muldiv, device_ctx.clone());
                inventory.add_executor_chip(muldiv);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

/// This builder will do tracegen for the RV64IM extensions on GPU but the modular extensions on
/// CPU.
#[derive(Clone)]
pub struct Rv64ModularHybridBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Rv64ModularHybridBuilder {
    type VmConfig = Rv64ModularConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv64ModularConfig,
        circuit: AirInventory<SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, GpuBackend, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
            device_ctx,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImGpuProverExt, &config.base, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImGpuProverExt, &config.mul, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv64ImGpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &AlgebraHybridProverExt,
            &config.modular,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

/// This builder will do tracegen for the RV64IM extensions on GPU but the modular and complex
/// extensions on CPU.
#[derive(Clone)]
pub struct Rv64ModularWithFp2HybridBuilder;

impl VmBuilder<E> for Rv64ModularWithFp2HybridBuilder {
    type VmConfig = Rv64ModularWithFp2Config;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv64ModularWithFp2Config,
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
            &AlgebraHybridProverExt,
            &config.fp2,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
