//! Prover extension for the GPU backend which still does trace generation on CPU.

use openvm_algebra_transpiler::Rv32ModularArithmeticOpcode;
use openvm_circuit::{
    arch::*,
    system::{
        cuda::{
            extensions::{
                get_inventory_range_checker, get_or_create_bitwise_op_lookup, SystemGpuBuilder,
            },
            SystemChipInventoryGPU,
        },
        memory::SharedMemoryHelper,
    },
};
use openvm_circuit_primitives::bigint::utils::big_uint_to_limbs;
use openvm_cuda_backend::{
    chip::{cpu_proving_ctx_to_gpu, get_empty_air_proving_ctx},
    engine::GpuBabyBearPoseidon2Engine,
    prover_backend::GpuBackend,
    types::{F, SC},
};
use openvm_instructions::LocalOpcode;
use openvm_mod_circuit_builder::{ExprBuilderConfig, FieldExpressionMetadata};
use openvm_rv32_adapters::{
    Rv32IsEqualModAdapterCols, Rv32IsEqualModAdapterExecutor, Rv32IsEqualModAdapterFiller,
    Rv32IsEqualModAdapterRecord, Rv32VecHeapAdapterCols, Rv32VecHeapAdapterExecutor,
};
use openvm_rv32im_circuit::Rv32ImGpuProverExt;
use openvm_stark_backend::{p3_air::BaseAir, prover::types::AirProvingContext, Chip};
use strum::EnumCount;

use crate::{
    fp2_chip::{get_fp2_addsub_chip, get_fp2_muldiv_chip, Fp2Air, Fp2Chip},
    modular_chip::*,
    AlgebraRecord, Fp2Extension, ModularExtension, Rv32ModularConfig, Rv32ModularWithFp2Config,
};

#[derive(derive_new::new)]
pub struct HybridModularChip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    cpu: ModularChip<F, BLOCKS, BLOCK_SIZE>,
}

// Auto-implementation of Chip for GpuBackend for a Cpu Chip by doing conversion
// of Dense->Matrix Record Arena, cpu tracegen, and then H2D transfer of the trace matrix.
impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for HybridModularChip<F, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size = RecordSeeker::<
            DenseRecordArena,
            AlgebraRecord<2, BLOCKS, BLOCK_SIZE>,
            _,
        >::get_aligned_record_size(&layout);

        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;

        let height = num_records.next_power_of_two();
        let mut seeker = arena
            .get_record_seeker::<AlgebraRecord<2, BLOCKS, BLOCK_SIZE>, AdapterCoreLayout<
                FieldExpressionMetadata<
                    F,
                    Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
                >,
            >>();
        let adapter_width =
            Rv32VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width();
        let width = adapter_width + BaseAir::<F>::width(&self.cpu.inner.expr);
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, layout);
        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_gpu(ctx)
    }
}

#[derive(derive_new::new)]
pub struct HybridModularIsEqualChip<
    F,
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> {
    cpu: ModularIsEqualChip<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
}

impl<const NUM_LANES: usize, const LANE_SIZE: usize, const TOTAL_LIMBS: usize>
    Chip<DenseRecordArena, GpuBackend>
    for HybridModularIsEqualChip<F, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>
{
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let record_size = size_of::<(
            Rv32IsEqualModAdapterRecord<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
            ModularIsEqualRecord<TOTAL_LIMBS>,
        )>();
        let trace_width = Rv32IsEqualModAdapterCols::<F, 2, NUM_LANES, LANE_SIZE>::width()
            + ModularIsEqualCoreCols::<F, TOTAL_LIMBS>::width();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let height = num_records.next_power_of_two();
        let mut seeker = arena.get_record_seeker::<(
            &mut Rv32IsEqualModAdapterRecord<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
            &mut ModularIsEqualRecord<TOTAL_LIMBS>,
        ), EmptyAdapterCoreLayout<
            F,
            Rv32IsEqualModAdapterExecutor<2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
        >>();
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, trace_width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, EmptyAdapterCoreLayout::new());
        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_gpu(ctx)
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
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let range_checker = range_checker_gpu.cpu_chip.clone().unwrap();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let bitwise_lu_gpu = get_or_create_bitwise_op_lookup(inventory)?;
        let bitwise_lu = bitwise_lu_gpu.cpu_chip.clone().unwrap();

        for (i, modulus) in extension.supported_moduli.iter().enumerate() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);
            let start_offset =
                Rv32ModularArithmeticOpcode::CLASS_OFFSET + i * Rv32ModularArithmeticOpcode::COUNT;

            let modulus_limbs = big_uint_to_limbs(modulus, 8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<1, 32>>()?;
                let addsub = get_modular_addsub_chip::<F, 1, 32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridModularChip::new(addsub));

                inventory.next_air::<ModularAir<1, 32>>()?;
                let muldiv = get_modular_muldiv_chip::<F, 1, 32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridModularChip::new(muldiv));

                let modulus_limbs = std::array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u8
                    } else {
                        0
                    }
                });
                inventory.next_air::<ModularIsEqualAir<1, 32, 32>>()?;
                let is_eq = ModularIsEqualChip::<F, 1, 32, 32>::new(
                    ModularIsEqualFiller::new(
                        Rv32IsEqualModAdapterFiller::new(pointer_max_bits, bitwise_lu.clone()),
                        start_offset,
                        modulus_limbs,
                        bitwise_lu.clone(),
                    ),
                    mem_helper.clone(),
                );
                inventory.add_executor_chip(HybridModularIsEqualChip::new(is_eq));
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<3, 16>>()?;
                let addsub = get_modular_addsub_chip::<F, 3, 16>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridModularChip::new(addsub));

                inventory.next_air::<ModularAir<3, 16>>()?;
                let muldiv = get_modular_muldiv_chip::<F, 3, 16>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridModularChip::new(muldiv));

                let modulus_limbs = std::array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u8
                    } else {
                        0
                    }
                });
                inventory.next_air::<ModularIsEqualAir<3, 16, 48>>()?;
                let is_eq = ModularIsEqualChip::<F, 3, 16, 48>::new(
                    ModularIsEqualFiller::new(
                        Rv32IsEqualModAdapterFiller::new(pointer_max_bits, bitwise_lu.clone()),
                        start_offset,
                        modulus_limbs,
                        bitwise_lu.clone(),
                    ),
                    mem_helper.clone(),
                );
                inventory.add_executor_chip(HybridModularIsEqualChip::new(is_eq));
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

#[derive(derive_new::new)]
pub struct HybridFp2Chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    cpu: Fp2Chip<F, BLOCKS, BLOCK_SIZE>,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for HybridFp2Chip<F, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, mut arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size = RecordSeeker::<
            DenseRecordArena,
            AlgebraRecord<2, BLOCKS, BLOCK_SIZE>,
            _,
        >::get_aligned_record_size(&layout);

        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % record_size, 0);

        let num_records = records.len() / record_size;
        let height = num_records.next_power_of_two();
        let mut seeker = arena
            .get_record_seeker::<AlgebraRecord<2, BLOCKS, BLOCK_SIZE>, AdapterCoreLayout<
                FieldExpressionMetadata<
                    F,
                    Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
                >,
            >>();
        let adapter_width =
            Rv32VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width();
        let width = adapter_width + BaseAir::<F>::width(&self.cpu.inner.expr);
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, layout);
        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_gpu(ctx)
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
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let range_checker = range_checker_gpu.cpu_chip.clone().unwrap();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let bitwise_lu_gpu = get_or_create_bitwise_op_lookup(inventory)?;
        let bitwise_lu = bitwise_lu_gpu.cpu_chip.clone().unwrap();

        for (_, modulus) in extension.supported_moduli.iter() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                inventory.next_air::<Fp2Air<2, 32>>()?;
                let addsub = get_fp2_addsub_chip::<F, 2, 32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridFp2Chip::new(addsub));

                inventory.next_air::<Fp2Air<2, 32>>()?;
                let muldiv = get_fp2_muldiv_chip::<F, 2, 32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridFp2Chip::new(muldiv));
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                inventory.next_air::<Fp2Air<6, 16>>()?;
                let addsub = get_fp2_addsub_chip::<F, 6, 16>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridFp2Chip::new(addsub));

                inventory.next_air::<Fp2Air<6, 16>>()?;
                let muldiv = get_fp2_muldiv_chip::<F, 6, 16>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridFp2Chip::new(muldiv));
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

/// This builder will do tracegen for the RV32IM extensions on GPU but the modular extensions on
/// CPU.
#[derive(Clone)]
pub struct Rv32ModularHybridBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Rv32ModularHybridBuilder {
    type VmConfig = Rv32ModularConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv32ModularConfig,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, GpuBackend, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemGpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.base, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.mul, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &AlgebraHybridProverExt,
            &config.modular,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

/// This builder will do tracegen for the RV32IM extensions on GPU but the modular and complex
/// extensions on CPU.
#[derive(Clone)]
pub struct Rv32ModularWithFp2HybridBuilder;

impl VmBuilder<E> for Rv32ModularWithFp2HybridBuilder {
    type VmConfig = Rv32ModularWithFp2Config;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv32ModularWithFp2Config,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, GpuBackend, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &Rv32ModularHybridBuilder,
            &config.modular,
            circuit,
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
