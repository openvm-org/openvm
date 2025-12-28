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
use openvm_instructions::LocalOpcode;
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use openvm_te_transpiler::Rv32EdwardsOpcode;
use strum::EnumCount;

use crate::{edwards_chip::TeAddChipGpu, EdwardsAir, EdwardsExtension, Rv32EdwardsConfig};

#[derive(Clone)]
pub struct EdwardsGpuProverExt;

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, EdwardsExtension>
    for EdwardsGpuProverExt
{
    fn extend_prover(
        &self,
        extension: &EdwardsExtension,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        // Range checker should always exist in inventory
        let range_checker = get_inventory_range_checker(inventory);

        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        for (i, curve) in extension.supported_curves.iter().enumerate() {
            let start_offset = Rv32EdwardsOpcode::CLASS_OFFSET + i * Rv32EdwardsOpcode::COUNT;
            let bytes = curve.modulus.bits().div_ceil(8);

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: curve.modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                inventory.next_air::<EdwardsAir<2, 2, 32>>()?;
                let add = TeAddChipGpu::<2, 32>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config.clone(),
                    curve.a.clone(),
                    curve.d.clone(),
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(add);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct Rv32EdwardsGpuBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Rv32EdwardsGpuBuilder {
    type VmConfig = Rv32EdwardsConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv32EdwardsConfig,
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
            &EdwardsGpuProverExt,
            &config.edwards,
            inventory,
        )?;

        Ok(chip_complex)
    }
}
