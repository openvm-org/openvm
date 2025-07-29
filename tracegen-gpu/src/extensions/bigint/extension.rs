use std::sync::Arc;

use openvm_bigint_circuit::{
    Int256, Int256Rv32Config, Rv32BaseAlu256Air, Rv32BranchEqual256Air, Rv32BranchLessThan256Air,
    Rv32LessThan256Air, Rv32Multiplication256Air, Rv32Shift256Air,
};
use openvm_circuit::arch::{
    AirInventory, ChipInventory, ChipInventoryError, DenseRecordArena, VmBuilder, VmChipComplex,
    VmProverExtension,
};
use openvm_circuit_primitives::range_tuple::RangeTupleCheckerAir;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use stark_backend_gpu::{engine::GpuBabyBearPoseidon2Engine, prover_backend::GpuBackend};

use crate::{
    extensions::{
        bigint::{
            BaseAlu256ChipGpu, BranchEqual256ChipGpu, BranchLessThan256ChipGpu, LessThan256ChipGpu,
            Multiplication256ChipGpu, Shift256ChipGpu,
        },
        rv32im::Rv32ImGpuProverExt,
    },
    primitives::range_tuple::RangeTupleCheckerChipGPU,
    system::{
        extensions::{
            get_inventory_range_checker, get_or_create_bitwise_op_lookup, SystemGpuBuilder,
        },
        SystemChipInventoryGPU,
    },
};

pub struct Int256GpuProverExt;

// This implementation is specific to GpuBackend because the lookup chips
// (VariableRangeCheckerChipGPU, BitwiseOperationLookupChipGPU) are specific to GpuBackend.
impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Int256>
    for Int256GpuProverExt
{
    fn extend_prover(
        &self,
        extension: &Int256,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let pointer_max_bits = inventory.airs().pointer_max_bits();
        let timestamp_max_bits = inventory.timestamp_max_bits();

        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<Arc<RangeTupleCheckerChipGPU<2>>>()
                .find(|c| {
                    c.sizes[0] >= extension.range_tuple_checker_sizes[0]
                        && c.sizes[1] >= extension.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                inventory.next_air::<RangeTupleCheckerAir<2>>()?;
                let chip = Arc::new(RangeTupleCheckerChipGPU::new(
                    extension.range_tuple_checker_sizes,
                ));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        // 1. alu (BaseAlu256)
        inventory.next_air::<Rv32BaseAlu256Air>()?;
        let base_alu = BaseAlu256ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(base_alu);

        // 2. lt (LessThan256)
        inventory.next_air::<Rv32LessThan256Air>()?;
        let lt = LessThan256ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(lt);

        // 3. beq (BranchEqual256)
        inventory.next_air::<Rv32BranchEqual256Air>()?;
        let beq = BranchEqual256ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(beq);

        // 4. blt (BranchLessThan256)
        inventory.next_air::<Rv32BranchLessThan256Air>()?;
        let blt = BranchLessThan256ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(blt);

        // 5. mult (Multiplication256)
        inventory.next_air::<Rv32Multiplication256Air>()?;
        let mult = Multiplication256ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            range_tuple_checker.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(mult);

        // 6. shift (Shift256)
        inventory.next_air::<Rv32Shift256Air>()?;
        let shift = Shift256ChipGpu::new(
            range_checker.clone(),
            bitwise_lu.clone(),
            pointer_max_bits,
            timestamp_max_bits,
        );
        inventory.add_executor_chip(shift);

        Ok(())
    }
}

type E = GpuBabyBearPoseidon2Engine;

#[derive(Clone)]
pub struct Int256Rv32GpuBuilder;

impl VmBuilder<GpuBabyBearPoseidon2Engine> for Int256Rv32GpuBuilder {
    type VmConfig = Int256Rv32Config;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Int256Rv32Config,
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
            &Int256GpuProverExt,
            &config.bigint,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
