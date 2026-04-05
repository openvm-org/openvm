use std::sync::Arc;

use openvm_circuit::{
    arch::{
        AirInventory, ChipInventory, ChipInventoryError, DenseRecordArena, VmBuilder,
        VmChipComplex, VmProverExtension,
    },
    system::cuda::{
        extensions::{
            get_inventory_range_checker, get_or_create_bitwise_op_lookup, SystemGpuBuilder,
        },
        SystemChipInventoryGPU,
    },
};
use openvm_cuda_backend::{
    prelude::F as CudaF, BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend,
};
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_rv32im_circuit::Rv32ImGpuProverExt;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::{
    call::{DeferralCallAir, DeferralCallChipGpu},
    count::{DeferralCircuitCountAir, DeferralCircuitCountChipGpu},
    output::{DeferralOutputAir, DeferralOutputChipGpu},
    poseidon2::{DeferralPoseidon2Air, DeferralPoseidon2ChipGpu},
    DeferralExtension, Rv32DeferralConfig,
};

pub struct DeferralGpuProverExt;

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, DeferralExtension>
    for DeferralGpuProverExt
{
    fn extend_prover(
        &self,
        extension: &DeferralExtension,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let num_deferral_circuits = extension.fns.len();
        let address_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        let count = Arc::new(if num_deferral_circuits == 0 {
            DeviceBuffer::<u32>::new()
        } else {
            DeviceBuffer::<u32>::with_capacity_on(num_deferral_circuits, &range_checker.ctx)
        });
        if num_deferral_circuits > 0 {
            count.fill_zero_on(&range_checker.ctx).unwrap();
        }

        inventory.next_air::<DeferralCircuitCountAir>()?;
        let count_chip = Arc::new(DeferralCircuitCountChipGpu::new(
            count.clone(),
            num_deferral_circuits,
        ));
        inventory.add_periphery_chip(count_chip);

        inventory.next_air::<DeferralPoseidon2Air<CudaF>>()?;
        let max_trace_height = inventory
            .config()
            .segmentation_config
            .limits
            .max_trace_height as usize;
        let poseidon2_chip = Arc::new(DeferralPoseidon2ChipGpu::new(
            max_trace_height.max(1),
            1,
            range_checker.ctx.clone(),
        ));
        let poseidon2_shared = poseidon2_chip.shared_buffer();
        inventory.add_periphery_chip(poseidon2_chip);

        inventory.next_air::<DeferralCallAir>()?;
        let call_chip = DeferralCallChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            address_bits,
            timestamp_max_bits,
            count.clone(),
            num_deferral_circuits,
            poseidon2_shared.clone(),
        );
        inventory.add_executor_chip(call_chip);

        inventory.next_air::<DeferralOutputAir>()?;
        let output_chip = DeferralOutputChipGpu::new(
            range_checker,
            bitwise_lu,
            address_bits,
            timestamp_max_bits,
            count,
            num_deferral_circuits,
            poseidon2_shared,
        );
        inventory.add_executor_chip(output_chip);

        Ok(())
    }
}

#[derive(Clone)]
pub struct Rv32DeferralGpuBuilder;

impl VmBuilder<GpuBabyBearPoseidon2Engine> for Rv32DeferralGpuBuilder {
    type VmConfig = Rv32DeferralConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
        device: &openvm_cuda_backend::GpuDevice,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            Self::RecordArena,
            GpuBackend,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<GpuBabyBearPoseidon2Engine>::create_chip_complex(
            &SystemGpuBuilder,
            &config.system,
            circuit,
            device,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Rv32ImGpuProverExt,
            &config.rv32i,
            inventory,
        )?;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Rv32ImGpuProverExt,
            &config.rv32m,
            inventory,
        )?;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Rv32ImGpuProverExt,
            &config.io,
            inventory,
        )?;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &DeferralGpuProverExt,
            &config.deferral,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
