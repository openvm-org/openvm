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
use openvm_cuda_backend::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};
use openvm_rv32im_circuit::Rv32ImGpuProverExt;
use openvm_sha2_air::{Sha256Config, Sha512Config};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use openvm_stark_sdk::engine::StarkEngine;

use super::*;
use crate::{
    cuda::{make_hybrid_chips, Sha2BlockHasherChipGpu, Sha2MainChipGpu},
    Sha2BlockHasherVmAir, Sha2MainAir,
};

pub struct Sha2GpuProverExt;

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Sha2> for Sha2GpuProverExt {
    fn extend_prover(
        &self,
        _: &Sha2,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker_gpu = get_inventory_range_checker(inventory);
        let bitwise_gpu = get_or_create_bitwise_op_lookup(inventory)?;

        let range_checker_cpu = range_checker_gpu
            .cpu_chip
            .clone()
            .expect("Hybrid SHA-2 GPU chips require hybrid range checker");
        let bitwise_cpu = bitwise_gpu
            .cpu_chip
            .clone()
            .expect("Hybrid SHA-2 GPU chips require hybrid bitwise lookup");

        // SHA-256
        inventory.next_air::<Sha2BlockHasherVmAir<Sha256Config>>()?;
        let (sha256_main_cpu, sha256_block_cpu, _sha256_records) =
            make_hybrid_chips::<Sha256Config>(
                range_checker_cpu.clone(),
                bitwise_cpu.clone(),
                pointer_max_bits,
                timestamp_max_bits,
            );
        let sha256_block_gpu = Sha2BlockHasherChipGpu::new(sha256_block_cpu);
        inventory.add_periphery_chip(sha256_block_gpu);

        inventory.next_air::<Sha2MainAir<Sha256Config>>()?;
        let sha256_main_gpu = Sha2MainChipGpu::new(sha256_main_cpu);
        inventory.add_executor_chip(sha256_main_gpu);

        // SHA-512 (also covers SHA-384 constraints)
        inventory.next_air::<Sha2BlockHasherVmAir<Sha512Config>>()?;
        let (sha512_main_cpu, sha512_block_cpu, _records) =
            make_hybrid_chips::<Sha512Config>(
                range_checker_cpu,
                bitwise_cpu,
                pointer_max_bits,
                timestamp_max_bits,
            );
        let sha512_block_gpu = Sha2BlockHasherChipGpu::new(sha512_block_cpu);
        inventory.add_periphery_chip(sha512_block_gpu);

        inventory.next_air::<Sha2MainAir<Sha512Config>>()?;
        let sha512_main_gpu = Sha2MainChipGpu::new(sha512_main_cpu);
        inventory.add_executor_chip(sha512_main_gpu);

        Ok(())
    }
}

pub struct Sha2Rv32GpuBuilder;

type E = GpuBabyBearPoseidon2Engine;

impl VmBuilder<E> for Sha2Rv32GpuBuilder {
    type VmConfig = Sha2Rv32Config;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Sha2Rv32Config,
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
        VmProverExtension::<E, _, _>::extend_prover(&Sha2GpuProverExt, &config.sha2, inventory)?;
        Ok(chip_complex)
    }
}
