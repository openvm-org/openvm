//! Prover extension for the GPU backend which still does trace generation on CPU.

use std::sync::Arc;

use openvm_circuit::{
    arch::{DEFAULT_BLOCK_SIZE, *},
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
use openvm_circuit_primitives::{
    bigint::utils::big_uint_to_limbs, bitwise_op_lookup::BitwiseOperationLookupChipGPU,
    var_range::VariableRangeCheckerChipGPU, Chip,
};
use openvm_cuda_backend::{
    prelude::{F, SC},
    BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend,
};
use openvm_mod_circuit_builder::{
    cuda::FieldExprChipGpu, ExprBuilderConfig, FieldExpressionMetadata,
};
use openvm_rv32_adapters::{Rv32VecHeapAdapterCols, Rv32VecHeapAdapterExecutor};
use openvm_rv32im_circuit::Rv32ImGpuProverExt;
use openvm_stark_backend::prover::AirProvingContext;

use crate::{
    cuda::ModularIsEqualChipGpu,
    fp2_chip::{get_fp2_addsub_chip, get_fp2_muldiv_chip, Fp2Air, Fp2Chip},
    modular_chip::*,
    AlgebraRecord, Fp2Extension, ModularExtension, Rv32ModularConfig, Rv32ModularWithFp2Config,
    FP2_BLOCKS_32, FP2_BLOCKS_48, MODULAR_BLOCKS_32, MODULAR_BLOCKS_48, NUM_LIMBS_32, NUM_LIMBS_48,
};

pub struct HybridModularChip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    gpu: FieldExprChipGpu,
    _phantom: std::marker::PhantomData<ModularChip<F, BLOCKS, BLOCK_SIZE>>,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> HybridModularChip<F, BLOCKS, BLOCK_SIZE> {
    pub fn new(
        cpu: ModularChip<F, BLOCKS, BLOCK_SIZE>,
        byte_ptr_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker_gpu: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup_gpu: Arc<BitwiseOperationLookupChipGPU<8>>,
    ) -> Self {
        let total_input_limbs = cpu.inner.num_inputs() * cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));
        let (adapter_size, core_size) = RecordSeeker::<
            DenseRecordArena,
            AlgebraRecord<2, BLOCKS, BLOCK_SIZE>,
            _,
        >::get_aligned_sizes(&layout);
        let gpu = FieldExprChipGpu::new(
            &cpu.inner,
            2,
            BLOCKS,
            Rv32VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width(),
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
impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for HybridModularChip<F, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        self.gpu.generate_proving_ctx(arena.allocated())
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
            let bytes = modulus.bits().div_ceil(8) as usize;

            let modulus_limbs = big_uint_to_limbs(modulus, 8);

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_32, DEFAULT_BLOCK_SIZE>>()?;
                let addsub = get_modular_addsub_chip::<F, MODULAR_BLOCKS_32, DEFAULT_BLOCK_SIZE>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridModularChip::new(
                    addsub,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                ));

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_32, DEFAULT_BLOCK_SIZE>>()?;
                let muldiv = get_modular_muldiv_chip::<F, MODULAR_BLOCKS_32, DEFAULT_BLOCK_SIZE>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridModularChip::new(
                    muldiv,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                ));

                let modulus_limbs = std::array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u8
                    } else {
                        0
                    }
                });
                inventory.next_air::<ModularIsEqualAir<MODULAR_BLOCKS_32, DEFAULT_BLOCK_SIZE, NUM_LIMBS_32>>()?;
                let is_eq = ModularIsEqualChipGpu::<
                    MODULAR_BLOCKS_32,
                    DEFAULT_BLOCK_SIZE,
                    NUM_LIMBS_32,
                >::new(
                    modulus_limbs,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                );
                inventory.add_executor_chip(is_eq);
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_48, DEFAULT_BLOCK_SIZE>>()?;
                let addsub = get_modular_addsub_chip::<F, MODULAR_BLOCKS_48, DEFAULT_BLOCK_SIZE>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridModularChip::new(
                    addsub,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                ));

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_48, DEFAULT_BLOCK_SIZE>>()?;
                let muldiv = get_modular_muldiv_chip::<F, MODULAR_BLOCKS_48, DEFAULT_BLOCK_SIZE>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridModularChip::new(
                    muldiv,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                ));

                let modulus_limbs = std::array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u8
                    } else {
                        0
                    }
                });
                inventory.next_air::<ModularIsEqualAir<MODULAR_BLOCKS_48, DEFAULT_BLOCK_SIZE, NUM_LIMBS_48>>()?;
                let is_eq = ModularIsEqualChipGpu::<
                    MODULAR_BLOCKS_48,
                    DEFAULT_BLOCK_SIZE,
                    NUM_LIMBS_48,
                >::new(
                    modulus_limbs,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                );
                inventory.add_executor_chip(is_eq);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

pub struct HybridFp2Chip<F, const BLOCKS: usize, const BLOCK_SIZE: usize> {
    gpu: FieldExprChipGpu,
    _phantom: std::marker::PhantomData<Fp2Chip<F, BLOCKS, BLOCK_SIZE>>,
}

impl<const BLOCKS: usize, const BLOCK_SIZE: usize> HybridFp2Chip<F, BLOCKS, BLOCK_SIZE> {
    pub fn new(
        cpu: Fp2Chip<F, BLOCKS, BLOCK_SIZE>,
        byte_ptr_max_bits: usize,
        timestamp_max_bits: usize,
        range_checker_gpu: Arc<VariableRangeCheckerChipGPU>,
        bitwise_lookup_gpu: Arc<BitwiseOperationLookupChipGPU<8>>,
    ) -> Self {
        let total_input_limbs = cpu.inner.num_inputs() * cpu.inner.expr.canonical_num_limbs();
        let layout = AdapterCoreLayout::with_metadata(FieldExpressionMetadata::<
            F,
            Rv32VecHeapAdapterExecutor<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >::new(total_input_limbs));
        let (adapter_size, core_size) = RecordSeeker::<
            DenseRecordArena,
            AlgebraRecord<2, BLOCKS, BLOCK_SIZE>,
            _,
        >::get_aligned_sizes(&layout);
        let gpu = FieldExprChipGpu::new(
            &cpu.inner,
            2,
            BLOCKS,
            Rv32VecHeapAdapterCols::<F, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>::width(),
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
impl<const BLOCKS: usize, const BLOCK_SIZE: usize> Chip<DenseRecordArena, GpuBackend>
    for HybridFp2Chip<F, BLOCKS, BLOCK_SIZE>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        self.gpu.generate_proving_ctx(arena.allocated())
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
            let bytes = modulus.bits().div_ceil(8) as usize;

            if bytes <= NUM_LIMBS_32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_32,
                    limb_bits: 8,
                };

                inventory.next_air::<Fp2Air<FP2_BLOCKS_32, DEFAULT_BLOCK_SIZE>>()?;
                let addsub = get_fp2_addsub_chip::<F, FP2_BLOCKS_32, DEFAULT_BLOCK_SIZE>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridFp2Chip::new(
                    addsub,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                ));

                inventory.next_air::<Fp2Air<FP2_BLOCKS_32, DEFAULT_BLOCK_SIZE>>()?;
                let muldiv = get_fp2_muldiv_chip::<F, FP2_BLOCKS_32, DEFAULT_BLOCK_SIZE>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridFp2Chip::new(
                    muldiv,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                ));
            } else if bytes <= NUM_LIMBS_48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: NUM_LIMBS_48,
                    limb_bits: 8,
                };

                inventory.next_air::<Fp2Air<FP2_BLOCKS_48, DEFAULT_BLOCK_SIZE>>()?;
                let addsub = get_fp2_addsub_chip::<F, FP2_BLOCKS_48, DEFAULT_BLOCK_SIZE>(
                    config.clone(),
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridFp2Chip::new(
                    addsub,
                    pointer_max_bits,
                    timestamp_max_bits,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                ));

                inventory.next_air::<Fp2Air<FP2_BLOCKS_48, DEFAULT_BLOCK_SIZE>>()?;
                let muldiv = get_fp2_muldiv_chip::<F, FP2_BLOCKS_48, DEFAULT_BLOCK_SIZE>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(HybridFp2Chip::new(
                    muldiv,
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
            &AlgebraHybridProverExt,
            &config.fp2,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
