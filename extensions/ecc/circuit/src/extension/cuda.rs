use openvm_algebra_circuit::Rv32ModularGpuBuilder;
use openvm_circuit::{
    arch::{
        AirInventory, ChipInventory, ChipInventoryError, DenseRecordArena, VmBuilder,
        VmChipComplex, VmProverExtension,
    },
    system::cuda::{
        extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
        SystemChipInventoryGPU,
    },
};
use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
use openvm_ecc_transpiler::Rv32WeierstrassOpcode;
use openvm_instructions::LocalOpcode;
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use strum::EnumCount;

use crate::{
    Rv32WeierstrassConfig, WeierstrassAddNeChipGpu, WeierstrassAir, WeierstrassDoubleChipGpu,
    WeierstrassExtension,
};

#[derive(Clone)]
pub struct EccGpuProverExt;

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, WeierstrassExtension>
    for EccGpuProverExt
{
    fn extend_prover(
        &self,
        extension: &WeierstrassExtension,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        // Range checker should always exist in inventory
        let range_checker = get_inventory_range_checker(inventory);

        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        for (i, curve) in extension.supported_curves.iter().enumerate() {
            let start_offset =
                Rv32WeierstrassOpcode::CLASS_OFFSET + i * Rv32WeierstrassOpcode::COUNT;
            let bytes = curve.modulus.bits().div_ceil(8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, 2, 32>>()?;
                let addne = WeierstrassAddNeChipGpu::<2, 32>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config.clone(),
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(addne);

                inventory.next_air::<WeierstrassAir<1, 2, 32>>()?;
                let double = WeierstrassDoubleChipGpu::<2, 32>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config,
                    start_offset,
                    curve.a.clone(),
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(double);
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                inventory.next_air::<WeierstrassAir<2, 6, 16>>()?;
                let addne = WeierstrassAddNeChipGpu::<6, 16>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config.clone(),
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(addne);

                inventory.next_air::<WeierstrassAir<1, 6, 16>>()?;
                let double = WeierstrassDoubleChipGpu::<6, 16>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config,
                    start_offset,
                    curve.a.clone(),
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(double);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

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
        circuit: AirInventory<BabyBearPoseidon2Config>,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            Self::RecordArena,
            GpuBackend,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&Rv32ModularGpuBuilder, &config.modular, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(
            &EccGpuProverExt,
            &config.weierstrass,
            inventory,
        )?;

        Ok(chip_complex)
    }
}
