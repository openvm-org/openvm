//! Prover extension for the GPU backend.

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
use openvm_circuit_primitives::bigint::utils::big_uint_to_limbs;
use openvm_cuda_backend::{
    prelude::{F, SC},
    BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend,
};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_rv32im_circuit::Rv32ImGpuProverExt;

use crate::{
    cuda::{
        Fp2AddSubChipGpu, Fp2MulDivChipGpu, ModularAddSubChipGpu, ModularIsEqualChipGpu,
        ModularMulDivChipGpu,
    },
    fp2_chip::{get_fp2_addsub_chip, get_fp2_muldiv_chip, Fp2Air},
    modular_chip::*,
    Fp2Extension, ModularExtension, Rv32ModularConfig, Rv32ModularWithFp2Config, FP2_BLOCKS_32,
    FP2_BLOCKS_48, MODULAR_BLOCKS_32, MODULAR_BLOCKS_48, NUM_LIMBS_32, NUM_LIMBS_48,
};

#[derive(Clone, Copy, Default)]
pub struct AlgebraGpuProverExt;

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, ModularExtension>
    for AlgebraGpuProverExt
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

        for modulus in &extension.supported_moduli {
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
                inventory.add_executor_chip(ModularAddSubChipGpu::new(
                    addsub,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
                ));

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_32, DEFAULT_BLOCK_SIZE>>()?;
                let muldiv = get_modular_muldiv_chip::<F, MODULAR_BLOCKS_32, DEFAULT_BLOCK_SIZE>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(ModularMulDivChipGpu::new(
                    muldiv,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
                ));

                let modulus_limbs = std::array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u8
                    } else {
                        0
                    }
                });
                inventory.next_air::<ModularIsEqualAir<MODULAR_BLOCKS_32, DEFAULT_BLOCK_SIZE, NUM_LIMBS_32>>()?;
                inventory.add_executor_chip(ModularIsEqualChipGpu::<
                    F,
                    MODULAR_BLOCKS_32,
                    DEFAULT_BLOCK_SIZE,
                    NUM_LIMBS_32,
                >::new(
                    modulus_limbs,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
                ));
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
                inventory.add_executor_chip(ModularAddSubChipGpu::new(
                    addsub,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
                ));

                inventory.next_air::<ModularAir<MODULAR_BLOCKS_48, DEFAULT_BLOCK_SIZE>>()?;
                let muldiv = get_modular_muldiv_chip::<F, MODULAR_BLOCKS_48, DEFAULT_BLOCK_SIZE>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(ModularMulDivChipGpu::new(
                    muldiv,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
                ));

                let modulus_limbs = std::array::from_fn(|i| {
                    if i < modulus_limbs.len() {
                        modulus_limbs[i] as u8
                    } else {
                        0
                    }
                });
                inventory.next_air::<ModularIsEqualAir<MODULAR_BLOCKS_48, DEFAULT_BLOCK_SIZE, NUM_LIMBS_48>>()?;
                inventory.add_executor_chip(ModularIsEqualChipGpu::<
                    F,
                    MODULAR_BLOCKS_48,
                    DEFAULT_BLOCK_SIZE,
                    NUM_LIMBS_48,
                >::new(
                    modulus_limbs,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
                ));
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Fp2Extension>
    for AlgebraGpuProverExt
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
                inventory.add_executor_chip(Fp2AddSubChipGpu::new(
                    addsub,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
                ));

                inventory.next_air::<Fp2Air<FP2_BLOCKS_32, DEFAULT_BLOCK_SIZE>>()?;
                let muldiv = get_fp2_muldiv_chip::<F, FP2_BLOCKS_32, DEFAULT_BLOCK_SIZE>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(Fp2MulDivChipGpu::new(
                    muldiv,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
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
                inventory.add_executor_chip(Fp2AddSubChipGpu::new(
                    addsub,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
                ));

                inventory.next_air::<Fp2Air<FP2_BLOCKS_48, DEFAULT_BLOCK_SIZE>>()?;
                let muldiv = get_fp2_muldiv_chip::<F, FP2_BLOCKS_48, DEFAULT_BLOCK_SIZE>(
                    config,
                    mem_helper.clone(),
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    pointer_max_bits,
                );
                inventory.add_executor_chip(Fp2MulDivChipGpu::new(
                    muldiv,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
                ));
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

/// This builder does tracegen for the RV32IM and algebra extensions on GPU.
#[derive(Clone)]
pub struct Rv32ModularGpuBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Rv32ModularGpuBuilder {
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
            &AlgebraGpuProverExt,
            &config.modular,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

/// This builder will do tracegen for the RV32IM extensions on GPU but the modular and complex
/// extensions on CPU.
#[derive(Clone)]
pub struct Rv32ModularWithFp2GpuBuilder;

impl VmBuilder<E> for Rv32ModularWithFp2GpuBuilder {
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
            &Rv32ModularGpuBuilder,
            &config.modular,
            circuit,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(
            &AlgebraGpuProverExt,
            &config.fp2,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
