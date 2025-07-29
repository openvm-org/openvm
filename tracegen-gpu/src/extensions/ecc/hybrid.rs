use openvm_circuit::{
    arch::{
        AdapterCoreLayout, Arena, ChipInventory, ChipInventoryError, DenseRecordArena,
        MatrixRecordArena, RecordSeeker, VmProverExtension,
    },
    system::memory::SharedMemoryHelper,
};
use openvm_ecc_circuit::{
    get_ec_addne_chip, get_ec_double_chip, WeierstrassAir, WeierstrassChip, WeierstrassExtension,
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
    extensions::ecc::EccRecord,
    get_empty_air_proving_ctx,
    system::extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
};

#[derive(derive_new::new)]
pub struct HybridWeierstrassChip<
    F,
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
> {
    pub cpu: WeierstrassChip<F, NUM_READS, BLOCKS, BLOCK_SIZE>,
}

// Auto-implementation of Chip for GpuBackend for a Cpu Chip by doing conversion
// of Dense->Matrix Record Arena, cpu tracegen, and then H2D transfer of the trace matrix.
impl<const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize>
    Chip<DenseRecordArena, GpuBackend> for HybridWeierstrassChip<F, NUM_READS, BLOCKS, BLOCK_SIZE>
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
            Rv32VecHeapAdapterStep<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));

        let record_size = RecordSeeker::<
            DenseRecordArena,
            EccRecord<NUM_READS, BLOCKS, BLOCK_SIZE>,
            _,
        >::get_aligned_record_size(&layout);
        let rows_used = dense_arena.allocated().len().div_ceil(record_size);
        let height = rows_used.next_power_of_two();
        let mut seeker = dense_arena
            .get_record_seeker::<EccRecord<NUM_READS, BLOCKS, BLOCK_SIZE>, AdapterCoreLayout<
                FieldExpressionMetadata<
                    F,
                    Rv32VecHeapAdapterStep<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
                >,
            >>();
        let adapter_width =
            Rv32VecHeapAdapterCols::<F, NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width();
        let width = adapter_width + BaseAir::<F>::width(&self.cpu.inner.expr);
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, layout);
        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_gpu(ctx)
    }
}

#[derive(Clone, Copy, Default)]
pub struct EccHybridProverExt;

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
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let range_checker = range_checker_gpu.cpu_chip.clone().unwrap();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

        let bitwise_lu_gpu = get_or_create_bitwise_op_lookup(inventory)?;
        let bitwise_lu = bitwise_lu_gpu.cpu_chip.clone().unwrap();

        for curve in extension.supported_curves.iter() {
            let bytes = curve.modulus.bits().div_ceil(8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, 2, 32>>()?;
                let addne = get_ec_addne_chip::<F, 2, 32>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridWeierstrassChip::new(addne));

                inventory.next_air::<WeierstrassAir<1, 2, 32>>()?;
                let double = get_ec_double_chip::<F, 2, 32>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                    curve.a.clone(),
                );
                inventory.add_executor_chip(HybridWeierstrassChip::new(double));
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, 6, 16>>()?;
                let addne = get_ec_addne_chip::<F, 6, 16>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridWeierstrassChip::new(addne));

                inventory.next_air::<WeierstrassAir<1, 6, 16>>()?;
                let double = get_ec_double_chip::<F, 6, 16>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                    curve.a.clone(),
                );
                inventory.add_executor_chip(HybridWeierstrassChip::new(double));
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}
