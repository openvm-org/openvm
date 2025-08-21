use openvm_algebra_transpiler::{Fp2Opcode, Rv32ModularArithmeticOpcode};
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
use openvm_instructions::LocalOpcode;
use openvm_mod_circuit_builder::ExprBuilderConfig;
use openvm_rv32im_circuit::Rv32ImGpuProverExt;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use strum::EnumCount;

use crate::{
    fp2_chip::{Fp2AddSubChipGpu, Fp2Air, Fp2MulDivChipGpu},
    modular_chip::{
        ModularAddSubChipGpu, ModularAir, ModularIsEqualAir, ModularIsEqualChipGpu,
        ModularMulDivChipGpu,
    },
    Fp2Extension, ModularExtension, Rv32ModularConfig, Rv32ModularWithFp2Config,
};

#[derive(Clone)]
pub struct AlgebraGpuProverExt;

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Fp2Extension>
    for AlgebraGpuProverExt
{
    fn extend_prover(
        &self,
        extension: &Fp2Extension,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        // Range checker should always exist in inventory
        let range_checker = get_inventory_range_checker(inventory);

        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        for (i, (_, modulus)) in extension.supported_moduli.iter().enumerate() {
            // Determine the number of bytes needed to represent a prime field element
            let bytes = modulus.bits().div_ceil(8);
            let start_offset = Fp2Opcode::CLASS_OFFSET + i * Fp2Opcode::COUNT;

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                inventory.next_air::<Fp2Air<2, 32>>()?;
                let addsub = Fp2AddSubChipGpu::<2, 32>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config.clone(),
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(addsub);

                inventory.next_air::<Fp2Air<2, 32>>()?;
                let muldiv = Fp2MulDivChipGpu::<2, 32>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config,
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(muldiv);
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                inventory.next_air::<Fp2Air<6, 16>>()?;
                let addsub = Fp2AddSubChipGpu::<6, 16>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config.clone(),
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(addsub);

                inventory.next_air::<Fp2Air<6, 16>>()?;
                let muldiv = Fp2MulDivChipGpu::<6, 16>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config,
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(muldiv);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, ModularExtension>
    for AlgebraGpuProverExt
{
    fn extend_prover(
        &self,
        extension: &ModularExtension,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        // Range checker should always exist in inventory
        let range_checker = get_inventory_range_checker(inventory);

        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        for (i, modulus) in extension.supported_moduli.iter().enumerate() {
            let bytes = modulus.bits().div_ceil(8);
            let start_offset =
                Rv32ModularArithmeticOpcode::CLASS_OFFSET + i * Rv32ModularArithmeticOpcode::COUNT;

            if bytes <= 32 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 32,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<1, 32>>()?;
                let addsub = ModularAddSubChipGpu::<1, 32>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config.clone(),
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(addsub);

                inventory.next_air::<ModularAir<1, 32>>()?;
                let muldiv = ModularMulDivChipGpu::<1, 32>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config,
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(muldiv);

                inventory.next_air::<ModularIsEqualAir<1, 32, 32>>()?;
                let is_eq = ModularIsEqualChipGpu::<1, 32, 32>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    modulus.clone(),
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(is_eq);
            } else if bytes <= 48 {
                let config = ExprBuilderConfig {
                    modulus: modulus.clone(),
                    num_limbs: 48,
                    limb_bits: 8,
                };

                inventory.next_air::<ModularAir<3, 16>>()?;
                let addsub = ModularAddSubChipGpu::<3, 16>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config.clone(),
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(addsub);

                inventory.next_air::<ModularAir<3, 16>>()?;
                let muldiv = ModularMulDivChipGpu::<3, 16>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    config,
                    start_offset,
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(muldiv);

                inventory.next_air::<ModularIsEqualAir<3, 16, 48>>()?;
                let is_eq = ModularIsEqualChipGpu::<3, 16, 48>::new(
                    range_checker.clone(),
                    bitwise_lu.clone(),
                    modulus.clone(),
                    pointer_max_bits as u32,
                    timestamp_max_bits as u32,
                );
                inventory.add_executor_chip(is_eq);
            } else {
                panic!("Modulus too large");
            }
        }

        Ok(())
    }
}

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

#[derive(Clone)]
pub struct Rv32ModularWithFp2GpuBuilder;

impl VmBuilder<E> for Rv32ModularWithFp2GpuBuilder {
    type VmConfig = Rv32ModularWithFp2Config;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Rv32ModularWithFp2Config,
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
        VmProverExtension::<E, _, _>::extend_prover(&AlgebraGpuProverExt, &config.fp2, inventory)?;
        Ok(chip_complex)
    }
}
