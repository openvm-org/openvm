use openvm_circuit::arch::{
    AirInventory, ChipInventory, ChipInventoryError, DenseRecordArena, VmBuilder, VmChipComplex,
    VmProverExtension,
};
use openvm_sha256_circuit::{Sha256, Sha256Rv32Config, Sha256VmAir};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_gpu::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};

use crate::{
    extensions::{rv32im::Rv32ImGpuProverExt, sha256::Sha256VmChipGpu},
    system::{
        extensions::{
            get_inventory_range_checker, get_or_create_bitwise_op_lookup, SystemGpuBuilder,
        },
        SystemChipInventoryGPU,
    },
};

pub struct Sha256GpuProverExt;

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Sha256>
    for Sha256GpuProverExt
{
    fn extend_prover(
        &self,
        _: &Sha256,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Sha256VmAir>()?;
        let sha256 = Sha256VmChipGpu::new(
            range_checker.clone(),
            bitwise_lu,
            pointer_max_bits as u32,
            timestamp_max_bits as u32,
        );
        inventory.add_executor_chip(sha256);

        Ok(())
    }
}

#[derive(Clone)]
pub struct Sha256Rv32GpuBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Sha256Rv32GpuBuilder {
    type VmConfig = Sha256Rv32Config;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Sha256Rv32Config,
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
            VmBuilder::<E>::create_chip_complex(&SystemGpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.rv32i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.rv32m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImGpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &Sha256GpuProverExt,
            &config.sha256,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
