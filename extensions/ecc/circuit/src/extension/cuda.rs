//! Prover extension for the GPU backend.

use openvm_algebra_circuit::Rv32ModularGpuBuilder;
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
use openvm_cuda_backend::prelude::{F, SC};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
use openvm_mod_circuit_builder::ExprBuilderConfig;
use crate::{
    cuda::{EcAddNeChipGpu, EcDoubleChipGpu},
    get_ec_addne_chip, get_ec_double_chip, Rv32WeierstrassConfig, WeierstrassAir,
    WeierstrassExtension, ECC_BLOCKS_32, ECC_BLOCKS_48, NUM_LIMBS_32, NUM_LIMBS_48,
};

#[derive(Clone, Copy, Default)]
pub struct EccGpuProverExt;

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, WeierstrassExtension>
    for EccGpuProverExt
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
                inventory.add_executor_chip(EcAddNeChipGpu::new(
                    addne,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
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
                inventory.add_executor_chip(EcDoubleChipGpu::new(
                    double,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
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
                inventory.add_executor_chip(EcAddNeChipGpu::new(
                    addne,
                    range_checker_gpu.clone(),
                    bitwise_lu_gpu.clone(),
                    pointer_max_bits,
                    timestamp_max_bits,
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
                inventory.add_executor_chip(EcDoubleChipGpu::new(
                    double,
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

/// This builder does tracegen for the RV32IM, modular, and ecc extensions on GPU.
#[derive(Clone)]
pub struct Rv32WeierstrassGpuBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Rv32WeierstrassGpuBuilder {
    type VmConfig = Rv32WeierstrassConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv32WeierstrassConfig,
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
            &EccGpuProverExt,
            &config.weierstrass,
            inventory,
        )?;

        Ok(chip_complex)
    }
}
