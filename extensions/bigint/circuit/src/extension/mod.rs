use std::sync::Arc;

use derive_more::derive::From;
use openvm_bigint_transpiler::{
    Rv32BaseAlu256Opcode, Rv32BranchEqual256Opcode, Rv32BranchLessThan256Opcode,
    Rv32LessThan256Opcode, Rv32Mul256Opcode, Rv32Shift256Opcode,
};
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, MatrixRecordArena, RowMajorMatrixArena,
        VmBuilder, VmChipComplex, VmCircuitExtension, VmExecutionExtension, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemChipInventory, SystemCpuBuilder, SystemPort},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    range_tuple::{
        RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip,
        SharedRangeTupleCheckerChip,
    },
};
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_rv32im_circuit::Rv32ImCpuProverExt;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use serde::{Deserialize, Serialize};

use crate::*;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use self::cuda::*;
        pub use self::cuda::{
            Int256GpuProverExt as Int256ProverExt,
            Int256Rv32GpuBuilder as Int256Rv32Builder,
        };
    } else {
        pub use self::{
            Int256CpuProverExt as Int256ProverExt,
            Int256Rv32CpuBuilder as Int256Rv32Builder,
        };
    }
}

// =================================== VM Extension Implementation =================================
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Int256 {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for Int256 {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: default_range_tuple_checker_sizes(),
        }
    }
}

fn default_range_tuple_checker_sizes() -> [u32; 2] {
    [1 << 8, 32 * (1 << 8)]
}

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum Int256Executor {
    BaseAlu256(Rv32BaseAlu256Executor),
    LessThan256(Rv32LessThan256Executor),
    BranchEqual256(Rv32BranchEqual256Executor),
    BranchLessThan256(Rv32BranchLessThan256Executor),
    Multiplication256(Rv32Multiplication256Executor),
    Shift256(Rv32Shift256Executor),
}

impl<F: PrimeField32> VmExecutionExtension<F> for Int256 {
    type Executor = Int256Executor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Int256Executor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();

        let alu = Rv32BaseAlu256Executor::new(
            Rv32HeapAdapterExecutor::new(pointer_max_bits),
            Rv32BaseAlu256Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(alu, Rv32BaseAlu256Opcode::iter().map(|x| x.global_opcode()))?;

        let lt = Rv32LessThan256Executor::new(
            Rv32HeapAdapterExecutor::new(pointer_max_bits),
            Rv32LessThan256Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(lt, Rv32LessThan256Opcode::iter().map(|x| x.global_opcode()))?;

        let beq = Rv32BranchEqual256Executor::new(
            Rv32HeapBranchAdapterExecutor::new(pointer_max_bits),
            Rv32BranchEqual256Opcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        );
        inventory.add_executor(
            beq,
            Rv32BranchEqual256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let blt = Rv32BranchLessThan256Executor::new(
            Rv32HeapBranchAdapterExecutor::new(pointer_max_bits),
            Rv32BranchLessThan256Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            blt,
            Rv32BranchLessThan256Opcode::iter().map(|x| x.global_opcode()),
        )?;

        let mult = Rv32Multiplication256Executor::new(
            Rv32HeapAdapterExecutor::new(pointer_max_bits),
            Rv32Mul256Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(mult, Rv32Mul256Opcode::iter().map(|x| x.global_opcode()))?;

        let shift = Rv32Shift256Executor::new(
            Rv32HeapAdapterExecutor::new(pointer_max_bits),
            Rv32Shift256Opcode::CLASS_OFFSET,
        );
        inventory.add_executor(shift, Rv32Shift256Opcode::iter().map(|x| x.global_opcode()))?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Int256 {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker = inventory.range_checker().bus;
        let pointer_max_bits = inventory.pointer_max_bits();

        let bitwise_lu = {
            // A trick to get around Rust's borrow rules
            let existing_air = inventory.find_air::<BitwiseOperationLookupAir<8>>().next();
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = BitwiseOperationLookupBus::new(inventory.new_bus_idx());
                let air = BitwiseOperationLookupAir::<8>::new(bus);
                inventory.add_air(air);
                air.bus
            }
        };

        let range_tuple_checker = {
            let existing_air = inventory.find_air::<RangeTupleCheckerAir<2>>().find(|c| {
                c.bus.sizes[0] >= self.range_tuple_checker_sizes[0]
                    && c.bus.sizes[1] >= self.range_tuple_checker_sizes[1]
            });
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = RangeTupleCheckerBus::new(
                    inventory.new_bus_idx(),
                    self.range_tuple_checker_sizes,
                );
                let air = RangeTupleCheckerAir { bus };
                inventory.add_air(air);
                air.bus
            }
        };

        let alu = Rv32BaseAlu256Air::new(
            Rv32HeapAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu, pointer_max_bits),
            BaseAluCoreAir::new(bitwise_lu, Rv32BaseAlu256Opcode::CLASS_OFFSET),
        );
        inventory.add_air(alu);

        let lt = Rv32LessThan256Air::new(
            Rv32HeapAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu, pointer_max_bits),
            LessThanCoreAir::new(bitwise_lu, Rv32LessThan256Opcode::CLASS_OFFSET),
        );
        inventory.add_air(lt);

        let beq = Rv32BranchEqual256Air::new(
            Rv32HeapBranchAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu, pointer_max_bits),
            BranchEqualCoreAir::new(Rv32BranchEqual256Opcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        );
        inventory.add_air(beq);

        let blt = Rv32BranchLessThan256Air::new(
            Rv32HeapBranchAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu, pointer_max_bits),
            BranchLessThanCoreAir::new(bitwise_lu, Rv32BranchLessThan256Opcode::CLASS_OFFSET),
        );
        inventory.add_air(blt);

        let mult = Rv32Multiplication256Air::new(
            Rv32HeapAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu, pointer_max_bits),
            MultiplicationCoreAir::new(range_tuple_checker, Rv32Mul256Opcode::CLASS_OFFSET),
        );
        inventory.add_air(mult);

        let shift = Rv32Shift256Air::new(
            Rv32HeapAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu, pointer_max_bits),
            ShiftCoreAir::new(bitwise_lu, range_checker, Rv32Shift256Opcode::CLASS_OFFSET),
        );
        inventory.add_air(shift);

        Ok(())
    }
}

pub struct Int256CpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Int256> for Int256CpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        extension: &Int256,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let pointer_max_bits = inventory.airs().config().memory_config.pointer_max_bits;

        let bitwise_lu = {
            let existing_chip = inventory
                .find_chip::<SharedBitwiseOperationLookupChip<8>>()
                .next();
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &BitwiseOperationLookupAir<8> = inventory.next_air()?;
                let chip = Arc::new(BitwiseOperationLookupChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        let range_tuple_checker = {
            let existing_chip = inventory
                .find_chip::<SharedRangeTupleCheckerChip<2>>()
                .find(|c| {
                    c.bus().sizes[0] >= extension.range_tuple_checker_sizes[0]
                        && c.bus().sizes[1] >= extension.range_tuple_checker_sizes[1]
                });
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &RangeTupleCheckerAir<2> = inventory.next_air()?;
                let chip = SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        inventory.next_air::<Rv32BaseAlu256Air>()?;
        let alu = Rv32BaseAlu256Chip::new(
            BaseAluFiller::new(
                Rv32HeapAdapterFiller::new(pointer_max_bits, bitwise_lu.clone()),
                bitwise_lu.clone(),
                Rv32BaseAlu256Opcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(alu);

        inventory.next_air::<Rv32LessThan256Air>()?;
        let lt = Rv32LessThan256Chip::new(
            LessThanFiller::new(
                Rv32HeapAdapterFiller::new(pointer_max_bits, bitwise_lu.clone()),
                bitwise_lu.clone(),
                Rv32LessThan256Opcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv32BranchEqual256Air>()?;
        let beq = Rv32BranchEqual256Chip::new(
            BranchEqualFiller::new(
                Rv32HeapBranchAdapterFiller::new(pointer_max_bits, bitwise_lu.clone()),
                Rv32BranchEqual256Opcode::CLASS_OFFSET,
                DEFAULT_PC_STEP,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv32BranchLessThan256Air>()?;
        let blt = Rv32BranchLessThan256Chip::new(
            BranchLessThanFiller::new(
                Rv32HeapBranchAdapterFiller::new(pointer_max_bits, bitwise_lu.clone()),
                bitwise_lu.clone(),
                Rv32BranchLessThan256Opcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv32Multiplication256Air>()?;
        let mult = Rv32Multiplication256Chip::new(
            MultiplicationFiller::new(
                Rv32HeapAdapterFiller::new(pointer_max_bits, bitwise_lu.clone()),
                range_tuple_checker.clone(),
                Rv32Mul256Opcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv32Shift256Air>()?;
        let shift = Rv32Shift256Chip::new(
            ShiftFiller::new(
                Rv32HeapAdapterFiller::new(pointer_max_bits, bitwise_lu.clone()),
                bitwise_lu.clone(),
                range_checker.clone(),
                Rv32Shift256Opcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift);
        Ok(())
    }
}

#[derive(Clone)]
pub struct Int256Rv32CpuBuilder;

impl<E, SC> VmBuilder<E> for Int256Rv32CpuBuilder
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: PrimeField32,
{
    type VmConfig = Int256Rv32Config;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &Int256Rv32Config,
        circuit: AirInventory<E::SC>,
    ) -> Result<
        VmChipComplex<E::SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemCpuBuilder, &config.system, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.rv32i, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.rv32m, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(&Rv32ImCpuProverExt, &config.io, inventory)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &Int256CpuProverExt,
            &config.bigint,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
