use openvm_algebra_circuit::{
    fp2_chip::{get_fp2_addsub_chip, get_fp2_muldiv_chip, Fp2Air, Fp2Chip},
    modular_chip::*,
    Fp2Extension, ModularExtension,
};
use openvm_circuit::{
    arch::{
        AdapterCoreLayout, Arena, ChipInventory, ChipInventoryError, DenseRecordArena,
        MatrixRecordArena, RecordSeeker, VmProverExtension,
    },
    system::memory::SharedMemoryHelper,
};
use openvm_mod_circuit_builder::{ExprBuilderConfig, FieldExpressionMetadata};
use openvm_rv32_adapters::{Rv32VecHeapAdapterCols, Rv32VecHeapAdapterStep};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use p3_air::BaseAir;
use stark_backend_gpu::{
    engine::GpuBabyBearPoseidon2Engine,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{
    cpu_proving_ctx_to_gpu,
    extensions::algebra::{AlgebraRecord, ModularIsEqualChipGpu},
    get_empty_air_proving_ctx,
    system::extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
};

#[derive(derive_new::new)]
pub struct HybridModularChip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub cpu: ModularChip<F, BLOCKS, BLOCK_SIZE>,
}

// Auto-implementation of Chip for GpuBackend for a Cpu Chip by doing conversion
// of Dense->Matrix Record Arena, cpu tracegen, and then H2D transfer of the trace matrix.
impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for HybridModularChip<F, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(
        &self,
        mut dense_arena: DenseRecordArena,
    ) -> AirProvingContext<GpuBackend> {
        if dense_arena.allocated().is_empty() {
            return get_empty_air_proving_ctx();
        }

        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterStep<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size = RecordSeeker::<
            DenseRecordArena,
            AlgebraRecord<2, BLOCKS, BLOCK_SIZE>,
            _,
        >::get_aligned_record_size(&layout);
        let rows_used = dense_arena.allocated().len().div_ceil(record_size);
        let height = rows_used.next_power_of_two();
        let mut seeker = dense_arena
            .get_record_seeker::<AlgebraRecord<2, BLOCKS, BLOCK_SIZE>, AdapterCoreLayout<
                FieldExpressionMetadata<
                    F,
                    Rv32VecHeapAdapterStep<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
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

        for modulus in extension.supported_moduli.iter() {
            // determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);

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

                inventory.next_air::<ModularIsEqualAir<1, 32, 32>>()?;
                let is_eq = ModularIsEqualChipGpu::<1, 32, 32>::new(
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    modulus.clone(),
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(is_eq);
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

                inventory.next_air::<ModularIsEqualAir<3, 16, 48>>()?;
                let is_eq = ModularIsEqualChipGpu::<3, 16, 48>::new(
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    modulus.clone(),
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(is_eq);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

#[derive(derive_new::new)]
pub struct HybridFp2Chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    pub cpu: Fp2Chip<F, BLOCKS, BLOCK_SIZE>,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for HybridFp2Chip<F, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(
        &self,
        mut dense_arena: DenseRecordArena,
    ) -> AirProvingContext<GpuBackend> {
        if dense_arena.allocated().is_empty() {
            return get_empty_air_proving_ctx();
        }

        let total_input_limbs =
            self.cpu.inner.num_inputs() * self.cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterStep<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size = RecordSeeker::<
            DenseRecordArena,
            AlgebraRecord<2, BLOCKS, BLOCK_SIZE>,
            _,
        >::get_aligned_record_size(&layout);
        let rows_used = dense_arena.allocated().len().div_ceil(record_size);
        let height = rows_used.next_power_of_two();
        let mut seeker = dense_arena
            .get_record_seeker::<AlgebraRecord<2, BLOCKS, BLOCK_SIZE>, AdapterCoreLayout<
                FieldExpressionMetadata<
                    F,
                    Rv32VecHeapAdapterStep<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
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
