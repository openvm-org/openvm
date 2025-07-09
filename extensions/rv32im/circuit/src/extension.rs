use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventory, ExecutorInventoryError, RowMajorMatrixArena, VmChipWrapper,
        VmCircuitExtension, VmExecutionExtension, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::AnyEnum;
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
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode, PhantomDiscriminant};
use openvm_rv32im_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode, LessThanOpcode,
    MulHOpcode, MulOpcode, Rv32AuipcOpcode, Rv32HintStoreOpcode, Rv32JalLuiOpcode, Rv32JalrOpcode,
    Rv32LoadStoreOpcode, Rv32Phantom, ShiftOpcode,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::cpu::CpuBackend,
};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{adapters::*, *};

// ============ Extension Struct Definitions ============

/// RISC-V 32-bit Base (RV32I) Extension
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv32I;

/// RISC-V Extension for handling IO (not to be confused with I base extension)
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv32Io;

/// RISC-V 32-bit Multiplication Extension (RV32M) Extension
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Rv32M {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for Rv32M {
    fn default() -> Self {
        Self {
            range_tuple_checker_sizes: default_range_tuple_checker_sizes(),
        }
    }
}

fn default_range_tuple_checker_sizes() -> [u32; 2] {
    [1 << 8, 8 * (1 << 8)]
}

// ============ Executor and Periphery Enums for Extension ============

/// RISC-V 32-bit Base (RV32I) Instruction Executors
#[derive(Clone, From, AnyEnum, InsExecutorE1, InstructionExecutor)]
pub enum Rv32IExecutor {
    // Rv32 (for standard 32-bit integers):
    BaseAlu(Rv32BaseAluStep),
    LessThan(Rv32LessThanStep),
    Shift(Rv32ShiftStep),
    LoadStore(Rv32LoadStoreStep),
    LoadSignExtend(Rv32LoadSignExtendStep),
    BranchEqual(Rv32BranchEqualStep),
    BranchLessThan(Rv32BranchLessThanStep),
    JalLui(Rv32JalLuiStep),
    Jalr(Rv32JalrStep),
    Auipc(Rv32AuipcStep),
}

/// RISC-V 32-bit Multiplication Extension (RV32M) Instruction Executors
#[derive(Clone, From, AnyEnum, InsExecutorE1, InstructionExecutor)]
pub enum Rv32MExecutor {
    Multiplication(Rv32MultiplicationStep),
    MultiplicationHigh(Rv32MulHStep),
    DivRem(Rv32DivRemStep),
}

/// RISC-V 32-bit Io Instruction Executors
#[derive(Clone, Copy, From, AnyEnum, InsExecutorE1, InstructionExecutor)]
pub enum Rv32IoExecutor {
    HintStore(Rv32HintStoreStep),
}

// ============ VmExtension Implementations ============

impl<F: PrimeField32> VmExecutionExtension<F> for Rv32I {
    type Executor = Rv32IExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventory<Rv32IExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        // TODO: [ExecutorInventory] needs to have pointer_max_bits
        let pointer_max_bits = 29;

        let base_alu = Rv32BaseAluStep::new(Rv32BaseAluAdapterStep, BaseAluOpcode::CLASS_OFFSET);
        inventory.add_executor(base_alu, BaseAluOpcode::iter().map(|x| x.global_opcode()))?;

        let lt = LessThanStep::new(Rv32BaseAluAdapterStep, LessThanOpcode::CLASS_OFFSET);
        inventory.add_executor(lt, LessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let shift = ShiftStep::new(Rv32BaseAluAdapterStep, ShiftOpcode::CLASS_OFFSET);
        inventory.add_executor(shift, ShiftOpcode::iter().map(|x| x.global_opcode()))?;

        let load_store = LoadStoreStep::new(
            Rv32LoadStoreAdapterStep::new(pointer_max_bits),
            Rv32LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            load_store,
            Rv32LoadStoreOpcode::iter()
                .take(Rv32LoadStoreOpcode::STOREB as usize + 1)
                .map(|x| x.global_opcode()),
        )?;

        let load_sign_extend =
            LoadSignExtendStep::new(Rv32LoadStoreAdapterStep::new(pointer_max_bits));
        inventory.add_executor(
            load_sign_extend,
            [Rv32LoadStoreOpcode::LOADB, Rv32LoadStoreOpcode::LOADH].map(|x| x.global_opcode()),
        )?;

        let beq = BranchEqualStep::new(
            Rv32BranchAdapterStep,
            BranchEqualOpcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        );
        inventory.add_executor(beq, BranchEqualOpcode::iter().map(|x| x.global_opcode()))?;

        let blt =
            BranchLessThanStep::new(Rv32BranchAdapterStep, BranchLessThanOpcode::CLASS_OFFSET);
        inventory.add_executor(blt, BranchLessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let jal_lui = Rv32JalLuiStep::new(Rv32CondRdWriteAdapterStep::new(Rv32RdWriteAdapterStep));
        inventory.add_executor(jal_lui, Rv32JalLuiOpcode::iter().map(|x| x.global_opcode()))?;

        let jalr = Rv32JalrStep::new(Rv32JalrAdapterStep);
        inventory.add_executor(jalr, Rv32JalrOpcode::iter().map(|x| x.global_opcode()))?;

        let auipc = Rv32AuipcStep::new(Rv32RdWriteAdapterStep);
        inventory.add_executor(auipc, Rv32AuipcOpcode::iter().map(|x| x.global_opcode()))?;

        // There is no downside to adding phantom sub-executors, so we do it in the base extension.
        inventory.add_phantom_sub_executor::<F, _>(
            phantom::Rv32HintInputSubEx,
            PhantomDiscriminant(Rv32Phantom::HintInput as u16),
        )?;
        inventory.add_phantom_sub_executor::<F, _>(
            phantom::Rv32HintRandomSubEx,
            PhantomDiscriminant(Rv32Phantom::HintRandom as u16),
        )?;
        inventory.add_phantom_sub_executor::<F, _>(
            phantom::Rv32PrintStrSubEx,
            PhantomDiscriminant(Rv32Phantom::PrintStr as u16),
        )?;
        inventory.add_phantom_sub_executor::<F, _>(
            phantom::Rv32HintLoadByKeySubEx,
            PhantomDiscriminant(Rv32Phantom::HintLoadByKey as u16),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Rv32I {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let range_checker = inventory.range_checker().bus;
        // TODO: add a getter for pointer_max_bits
        let pointer_max_bits = inventory.config().memory_config.pointer_max_bits;

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

        let base_alu = Rv32BaseAluAir::new(
            Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            BaseAluCoreAir::new(bitwise_lu, BaseAluOpcode::CLASS_OFFSET),
        );
        inventory.add_air(base_alu);

        let lt = Rv32LessThanAir::new(
            Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            LessThanCoreAir::new(bitwise_lu, LessThanOpcode::CLASS_OFFSET),
        );
        inventory.add_air(lt);

        let shift = Rv32ShiftAir::new(
            Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            ShiftCoreAir::new(bitwise_lu, range_checker, ShiftOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift);

        let load_store = Rv32LoadStoreAir::new(
            Rv32LoadStoreAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                pointer_max_bits,
            ),
            LoadStoreCoreAir::new(Rv32LoadStoreOpcode::CLASS_OFFSET),
        );
        inventory.add_air(load_store);

        let load_sign_extend = Rv32LoadSignExtendAir::new(
            Rv32LoadStoreAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                pointer_max_bits,
            ),
            LoadSignExtendCoreAir::new(range_checker),
        );
        inventory.add_air(load_sign_extend);

        let beq = Rv32BranchEqualAir::new(
            Rv32BranchAdapterAir::new(exec_bridge, memory_bridge),
            BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        );
        inventory.add_air(beq);

        let blt = Rv32BranchLessThanAir::new(
            Rv32BranchAdapterAir::new(exec_bridge, memory_bridge),
            BranchLessThanCoreAir::new(bitwise_lu, BranchLessThanOpcode::CLASS_OFFSET),
        );
        inventory.add_air(blt);

        let jal_lui = Rv32JalLuiAir::new(
            Rv32CondRdWriteAdapterAir::new(Rv32RdWriteAdapterAir::new(memory_bridge, exec_bridge)),
            Rv32JalLuiCoreAir::new(bitwise_lu),
        );
        inventory.add_air(jal_lui);

        let jalr = Rv32JalrAir::new(
            Rv32JalrAdapterAir::new(memory_bridge, exec_bridge),
            Rv32JalrCoreAir::new(bitwise_lu, range_checker),
        );
        inventory.add_air(jalr);

        let auipc = Rv32AuipcAir::new(
            Rv32RdWriteAdapterAir::new(memory_bridge, exec_bridge),
            Rv32AuipcCoreAir::new(bitwise_lu),
        );
        inventory.add_air(auipc);

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<SC, RA> VmProverExtension<SC, RA, CpuBackend<SC>> for Rv32I
where
    SC: StarkGenericConfig,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.airs().config().memory_config.clk_max_bits;
        // TODO: add a getter for pointer_max_bits
        let pointer_max_bits = 29;
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
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

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv32BaseAluAir>()?;
        let base_alu = Rv32BaseAluChip::new(
            BaseAluFiller::new(
                Rv32BaseAluAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                BaseAluOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(base_alu);

        inventory.next_air::<Rv32LessThanAir>()?;
        let lt = Rv32LessThanChip::new(
            LessThanFiller::new(
                Rv32BaseAluAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                LessThanOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv32ShiftAir>()?;
        let shift = Rv32ShiftChip::new(
            ShiftFiller::new(
                Rv32BaseAluAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                range_checker.clone(),
                ShiftOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift);

        inventory.next_air::<Rv32LoadStoreAir>()?;
        let load_store_chip = Rv32LoadStoreChip::new(
            LoadStoreFiller::new(
                Rv32LoadStoreAdapterFiller::new(pointer_max_bits, range_checker.clone()),
                Rv32LoadStoreOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_store_chip);

        inventory.next_air::<Rv32LoadSignExtendAir>()?;
        let load_sign_extend = Rv32LoadSignExtendChip::new(
            LoadSignExtendFiller::new(
                Rv32LoadStoreAdapterFiller::new(pointer_max_bits, range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_sign_extend);

        inventory.next_air::<Rv32BranchEqualAir>()?;
        let beq = Rv32BranchEqualChip::new(
            BranchEqualFiller::new(
                Rv32BranchAdapterFiller,
                BranchEqualOpcode::CLASS_OFFSET,
                DEFAULT_PC_STEP,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv32BranchLessThanAir>()?;
        let blt = Rv32BranchLessThanChip::new(
            BranchLessThanFiller::new(
                Rv32BranchAdapterFiller,
                bitwise_lu.clone(),
                BranchLessThanOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv32JalLuiAir>()?;
        let jal_lui = VmChipWrapper::new(
            Rv32JalLuiFiller::new(
                Rv32CondRdWriteAdapterFiller::new(Rv32RdWriteAdapterFiller),
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(jal_lui);

        inventory.next_air::<Rv32JalrAir>()?;
        let jalr = VmChipWrapper::new(
            Rv32JalrFiller::new(
                Rv32JalrAdapterFiller,
                bitwise_lu.clone(),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(jalr);

        inventory.next_air::<Rv32AuipcAir>()?;
        let auipc = VmChipWrapper::new(
            Rv32AuipcFiller::new(Rv32RdWriteAdapterFiller, bitwise_lu.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(auipc);

        Ok(())
    }
}

impl<F> VmExecutionExtension<F> for Rv32M {
    type Executor = Rv32MExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventory<Rv32MExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let mult = Rv32MultiplicationStep::new(Rv32MultAdapterStep, MulOpcode::CLASS_OFFSET);
        inventory.add_executor(mult, MulOpcode::iter().map(|x| x.global_opcode()))?;

        let mul_h = Rv32MulHStep::new(Rv32MultAdapterStep, MulHOpcode::CLASS_OFFSET);
        inventory.add_executor(mul_h, MulHOpcode::iter().map(|x| x.global_opcode()))?;

        let div_rem = Rv32DivRemStep::new(Rv32MultAdapterStep, DivRemOpcode::CLASS_OFFSET);
        inventory.add_executor(div_rem, DivRemOpcode::iter().map(|x| x.global_opcode()))?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Rv32M {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();
        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);

        let bitwise_lu = {
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

        let mult = Rv32MultiplicationAir::new(
            Rv32MultAdapterAir::new(exec_bridge, memory_bridge),
            MultiplicationCoreAir::new(range_tuple_checker, MulOpcode::CLASS_OFFSET),
        );
        inventory.add_air(mult);

        let mul_h = Rv32MulHAir::new(
            Rv32MultAdapterAir::new(exec_bridge, memory_bridge),
            MulHCoreAir::new(bitwise_lu, range_tuple_checker),
        );
        inventory.add_air(mul_h);

        let div_rem = Rv32DivRemAir::new(
            Rv32MultAdapterAir::new(exec_bridge, memory_bridge),
            DivRemCoreAir::new(bitwise_lu, range_tuple_checker, DivRemOpcode::CLASS_OFFSET),
        );
        inventory.add_air(div_rem);

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<SC, RA> VmProverExtension<SC, RA, CpuBackend<SC>> for Rv32M
where
    SC: StarkGenericConfig,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.airs().config().memory_config.clk_max_bits;
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

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
                    c.bus().sizes[0] >= self.range_tuple_checker_sizes[0]
                        && c.bus().sizes[1] >= self.range_tuple_checker_sizes[1]
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

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv32MultiplicationAir>()?;
        let mult = Rv32MultiplicationChip::new(
            MultiplicationFiller::new(
                Rv32MultAdapterFiller,
                range_tuple_checker.clone(),
                MulOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv32MulHAir>()?;
        let mul_h = Rv32MulHChip::new(
            MulHFiller::new(
                Rv32MultAdapterFiller,
                bitwise_lu.clone(),
                range_tuple_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(mul_h);

        inventory.next_air::<Rv32DivRemAir>()?;
        let div_rem = Rv32DivRemChip::new(
            DivRemFiller::new(
                Rv32MultAdapterFiller,
                bitwise_lu.clone(),
                range_tuple_checker.clone(),
                DivRemOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(div_rem);

        Ok(())
    }
}

impl<F> VmExecutionExtension<F> for Rv32Io {
    type Executor = Rv32IoExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventory<Rv32IoExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        // TODO: getter for pointer_max_bits
        let pointer_max_bits = 29;
        let hint_store =
            Rv32HintStoreStep::new(pointer_max_bits, Rv32HintStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            hint_store,
            Rv32HintStoreOpcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Rv32Io {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        // TODO: add a getter for pointer_max_bits
        let pointer_max_bits = inventory.config().memory_config.pointer_max_bits;

        let bitwise_lu = {
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

        let hint_store = Rv32HintStoreAir::new(
            exec_bridge,
            memory_bridge,
            bitwise_lu,
            Rv32HintStoreOpcode::CLASS_OFFSET,
            pointer_max_bits,
        );
        inventory.add_air(hint_store);

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<SC, RA> VmProverExtension<SC, RA, CpuBackend<SC>> for Rv32Io
where
    SC: StarkGenericConfig,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.airs().config().memory_config.clk_max_bits;
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        // TODO: add a getter for pointer_max_bits
        let pointer_max_bits = 29;

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

        inventory.next_air::<Rv32HintStoreAir>()?;
        let hint_store = VmChipWrapper::new(
            Rv32HintStoreFiller::new(pointer_max_bits, bitwise_lu.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(hint_store);

        Ok(())
    }
}

/// Phantom sub-executors
mod phantom {
    use eyre::bail;
    use openvm_circuit::{
        arch::{PhantomSubExecutor, Streams},
        system::memory::online::GuestMemory,
    };
    use openvm_instructions::PhantomDiscriminant;
    use openvm_stark_backend::p3_field::{Field, PrimeField32};
    use rand::{rngs::StdRng, Rng};

    use crate::adapters::{memory_read, read_rv32_register};

    pub struct Rv32HintInputSubEx;
    pub struct Rv32HintRandomSubEx;
    pub struct Rv32PrintStrSubEx;
    pub struct Rv32HintLoadByKeySubEx;

    impl<F: Field> PhantomSubExecutor<F> for Rv32HintInputSubEx {
        fn phantom_execute(
            &self,
            _: &GuestMemory,
            streams: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            _: u32,
            _: u32,
            _: u16,
        ) -> eyre::Result<()> {
            let mut hint = match streams.input_stream.pop_front() {
                Some(hint) => hint,
                None => {
                    bail!("EndOfInputStream");
                }
            };
            streams.hint_stream.clear();
            streams.hint_stream.extend(
                (hint.len() as u32)
                    .to_le_bytes()
                    .iter()
                    .map(|b| F::from_canonical_u8(*b)),
            );
            // Extend by 0 for 4 byte alignment
            let capacity = hint.len().div_ceil(4) * 4;
            hint.resize(capacity, F::ZERO);
            streams.hint_stream.extend(hint);
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv32HintRandomSubEx {
        fn phantom_execute(
            &self,
            memory: &GuestMemory,
            streams: &mut Streams<F>,
            rng: &mut StdRng,
            _: PhantomDiscriminant,
            a: u32,
            _: u32,
            _: u16,
        ) -> eyre::Result<()> {
            let len = read_rv32_register(memory, a) as usize;
            streams.hint_stream.clear();
            streams.hint_stream.extend(
                std::iter::repeat_with(|| F::from_canonical_u8(rng.gen::<u8>())).take(len * 4),
            );
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv32PrintStrSubEx {
        fn phantom_execute(
            &self,
            memory: &GuestMemory,
            _: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            a: u32,
            b: u32,
            _: u16,
        ) -> eyre::Result<()> {
            let rd = read_rv32_register(memory, a);
            let rs1 = read_rv32_register(memory, b);
            let bytes = (0..rs1)
                .map(|i| memory_read::<1>(memory, 2, rd + i)[0])
                .collect::<Vec<u8>>();
            let peeked_str = String::from_utf8(bytes)?;
            print!("{peeked_str}");
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv32HintLoadByKeySubEx {
        fn phantom_execute(
            &self,
            memory: &GuestMemory,
            streams: &mut Streams<F>,
            _: &mut StdRng,
            _: PhantomDiscriminant,
            a: u32,
            b: u32,
            _: u16,
        ) -> eyre::Result<()> {
            let ptr = read_rv32_register(memory, a);
            let len = read_rv32_register(memory, b);
            let key: Vec<u8> = (0..len)
                .map(|i| memory_read::<1>(memory, 2, ptr + i)[0])
                .collect();
            if let Some(val) = streams.kv_store.get(&key) {
                let to_push = hint_load_by_key_decode::<F>(val);
                for input in to_push.into_iter().rev() {
                    streams.input_stream.push_front(input);
                }
            } else {
                bail!("Rv32HintLoadByKey: key not found");
            }
            Ok(())
        }
    }

    pub fn hint_load_by_key_decode<F: PrimeField32>(value: &[u8]) -> Vec<Vec<F>> {
        let mut offset = 0;
        let len = extract_u32(value, offset) as usize;
        offset += 4;
        let mut ret = Vec::with_capacity(len);
        for _ in 0..len {
            let v_len = extract_u32(value, offset) as usize;
            offset += 4;
            let v = (0..v_len)
                .map(|_| {
                    let ret = F::from_canonical_u32(extract_u32(value, offset));
                    offset += 4;
                    ret
                })
                .collect();
            ret.push(v);
        }
        ret
    }

    fn extract_u32(value: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(value[offset..offset + 4].try_into().unwrap())
    }
}
