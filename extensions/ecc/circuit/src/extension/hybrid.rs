//! Prover extension for the GPU backend; FieldExpr chips do trace generation on GPU.

use std::sync::Arc;

use openvm_algebra_circuit::Rv32ModularHybridBuilder;
use openvm_circuit::{
    arch::{DEFAULT_BLOCK_SIZE, *},
    system::{
        cuda::{
            extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
            SystemChipInventoryGPU,
        },
        memory::SharedMemoryHelper,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{
    prelude::{F, SC},
    BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend,
};
use openvm_mod_circuit_builder::{
    cuda::FieldExprChipGpu, ExprBuilderConfig, FieldExpressionMetadata,
};
use openvm_rv32_adapters::{Rv32VecHeapAdapterCols, Rv32VecHeapAdapterExecutor};
use openvm_stark_backend::prover::AirProvingContext;

use crate::{
    get_ec_addne_chip, get_ec_double_chip, EccRecord, Rv32WeierstrassConfig, WeierstrassAir,
    WeierstrassChip, WeierstrassExtension, ECC_BLOCKS_32, ECC_BLOCKS_48, NUM_LIMBS_32,
    NUM_LIMBS_48,
};

pub struct HybridWeierstrassChip<
    F,
    const NUM_READS: usize,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
> {
    gpu: FieldExprChipGpu,
    _phantom: std::marker::PhantomData<WeierstrassChip<F, NUM_READS, BLOCKS, BLOCK_SIZE>>,
}

impl<const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize>
    HybridWeierstrassChip<F, NUM_READS, BLOCKS, BLOCK_SIZE>
{
    pub fn new(
        cpu: WeierstrassChip<F, NUM_READS, BLOCKS, BLOCK_SIZE>,
        byte_ptr_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker_gpu: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup_gpu: Arc<BitwiseOperationLookupChipGPU<8>>,
    ) -> Self {
        let total_input_limbs = cpu.inner.num_inputs() * cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterExecutor<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));
        let (adapter_size, core_size) = RecordSeeker::<
            DenseRecordArena,
            EccRecord<NUM_READS, BLOCKS, BLOCK_SIZE>,
            _,
        >::get_aligned_sizes(&layout);
        let gpu = FieldExprChipGpu::new(
            &cpu.inner,
            NUM_READS,
            BLOCKS,
            Rv32VecHeapAdapterCols::<F, NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width(),
            adapter_size + core_size,
            adapter_size,
            byte_ptr_max_bits as u32,
            timestamp_max_bits as u32,
            range_checker_gpu,
            bitwise_lookup_gpu,
        );
        Self {
            gpu,
            _phantom: std::marker::PhantomData,
        }
    }
}

// GPU tracegen: the field_expr kernel fills adapter + core columns directly from
// the dense records (see openvm_mod_circuit_builder::cuda).
impl<const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize>
    Chip<DenseRecordArena, GpuBackend> for HybridWeierstrassChip<F, NUM_READS, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        self.gpu.generate_proving_ctx(arena.allocated())
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
            let bytes = curve.modulus.bits().div_ceil(8) as usize;

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, ECC_BLOCKS_32, DEFAULT_BLOCK_SIZE>>()?;
                let addne = get_ec_addne_chip::<F, ECC_BLOCKS_32, DEFAULT_BLOCK_SIZE>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridWeierstrassChip::new(
                    addne,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                ));

                inventory.next_air::<WeierstrassAir<1, ECC_BLOCKS_32, DEFAULT_BLOCK_SIZE>>()?;
                let double = get_ec_double_chip::<F, ECC_BLOCKS_32, DEFAULT_BLOCK_SIZE>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                    curve.a.clone(),
                );
                inventory.add_executor_chip(HybridWeierstrassChip::new(
                    double,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                ));
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, ECC_BLOCKS_48, DEFAULT_BLOCK_SIZE>>()?;
                let addne = get_ec_addne_chip::<F, ECC_BLOCKS_48, DEFAULT_BLOCK_SIZE>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridWeierstrassChip::new(
                    addne,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                ));

                inventory.next_air::<WeierstrassAir<1, ECC_BLOCKS_48, DEFAULT_BLOCK_SIZE>>()?;
                let double = get_ec_double_chip::<F, ECC_BLOCKS_48, DEFAULT_BLOCK_SIZE>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                    curve.a.clone(),
                );
                inventory.add_executor_chip(HybridWeierstrassChip::new(
                    double,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                ));
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

/// This builder will do tracegen for the RV32IM extensions on GPU but the modular and ecc
/// extensions on CPU.
#[derive(Clone)]
pub struct Rv32WeierstrassHybridBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Rv32WeierstrassHybridBuilder {
    type VmConfig = Rv32WeierstrassConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv32WeierstrassConfig,
        circuit: AirInventory<SC>,
        device_ctx: &openvm_stark_backend::EngineDeviceCtx<E>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, GpuBackend, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<E>::create_chip_complex(
            &Rv32ModularHybridBuilder,
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
