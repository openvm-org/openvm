use std::sync::{Arc, Mutex};

use openvm_circuit::{
    arch::DenseRecordArena,
    system::cuda::{
        extensions::{
            get_inventory_range_checker, get_or_create_bitwise_op_lookup, SystemGpuBuilder,
        },
        SystemChipInventoryGPU,
    },
};
use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
use openvm_rv32im_circuit::Rv32ImGpuProverExt;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use super::*;
use crate::{
    cuda::{KeccakfOpChipGpu, KeccakfPermChipGpu, SharedKeccakfRecords, XorinVmChipGpu},
    keccakf_perm::KeccakfPermAir,
};

pub struct NewKeccak256GpuProverExt;

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, NewKeccak256>
    for NewKeccak256GpuProverExt
{
    fn extend_prover(
        &self,
        _extension: &NewKeccak256,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        // XorinVmChip
        inventory.next_air::<XorinVmAir>()?;
        let xorin_chip = XorinVmChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits as u32,
        );
        inventory.add_executor_chip(xorin_chip);

        // Create shared state for passing records between Op and Perm chips
        let shared_records = Arc::new(Mutex::new(SharedKeccakfRecords::default()));

        // NOTE: AIRs are added in extend_circuit in this order: XorinVmAir, KeccakfPermAir, KeccakfOpAir
        // The prover extension must consume AIRs in the same order.

        // Register KeccakfPermChip (periphery chip - added BEFORE OpChip to ensure OpChip tracegen runs first)
        inventory.next_air::<KeccakfPermAir>()?;
        let perm_chip = KeccakfPermChipGpu::new(shared_records.clone());
        inventory.add_periphery_chip(perm_chip);

        // Register KeccakfOpChip (executor chip - generates first due to executor vs periphery ordering)
        inventory.next_air::<KeccakfOpAir>()?;
        let op_chip = KeccakfOpChipGpu::new(
            range_checker,
            bitwise_lu,
            pointer_max_bits,
            timestamp_max_bits as u32,
            shared_records,
        );
        inventory.add_executor_chip(op_chip);

        Ok(())
    }
}

#[derive(Clone)]
pub struct NewKeccak256Rv32GpuBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for NewKeccak256Rv32GpuBuilder {
    type VmConfig = NewKeccak256Rv32Config;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &NewKeccak256Rv32Config,
        circuit: AirInventory<<E as StarkEngine>::SC>,
    ) -> Result<
        VmChipComplex<
            <E as StarkEngine>::SC,
            Self::RecordArena,
            <E as StarkEngine>::PB,
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
            &NewKeccak256GpuProverExt,
            &config.keccak,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
