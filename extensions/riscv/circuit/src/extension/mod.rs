use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, RowMajorMatrixArena, VmCircuitExtension,
        VmExecutionExtension, VmProverExtension,
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
use openvm_riscv_transpiler::{
    BaseAluOpcode, BranchEqualOpcode, BranchLessThanOpcode, DivRemOpcode, LessThanOpcode,
    MulHOpcode, MulOpcode, Rv64AuipcOpcode, Rv64HintStoreOpcode, Rv64JalLuiOpcode, Rv64JalrOpcode,
    Rv64LoadStoreOpcode, Rv64Phantom, ShiftOpcode,
};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    p3_field::PrimeField32,
    prover::cpu::{CpuBackend, CpuDevice},
};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{adapters::*, *};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use cuda::{
            Rv64ImGpuProverExt as Rv64ImGpuProverExt,
        };
    } else {
        pub use self::{
            Rv64ImCpuProverExt as Rv64ImProverExt,
        };
    }
}

// ============ Extension Struct Definitions ============

/// RISC-V 32-bit Base (RV32I) Extension
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv64I;

/// RISC-V Extension for handling IO (not to be confused with I base extension)
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv64Io;

/// RISC-V 32-bit Multiplication Extension (RV32M) Extension
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Rv64M {
    #[serde(default = "default_range_tuple_checker_sizes")]
    pub range_tuple_checker_sizes: [u32; 2],
}

impl Default for Rv64M {
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
// ITS THIS DERIVES FAULT; not supporting aot traits?
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum Rv64IExecutor {
    // Rv64 (for standard 32-bit integers):
    BaseAlu(Rv64BaseAluExecutor),
    LessThan(Rv64LessThanExecutor),
    Shift(Rv64ShiftExecutor),
    LoadStore(Rv64LoadStoreExecutor),
    LoadSignExtend(Rv64LoadSignExtendExecutor),
    BranchEqual(Rv64BranchEqualExecutor),
    BranchLessThan(Rv64BranchLessThanExecutor),
    JalLui(Rv64JalLuiExecutor),
    Jalr(Rv64JalrExecutor),
    Auipc(Rv64AuipcExecutor),
}

/// RISC-V 32-bit Multiplication Extension (RV32M) Instruction Executors
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum Rv64MExecutor {
    Multiplication(Rv64MultiplicationExecutor),
    MultiplicationHigh(Rv64MulHExecutor),
    DivRem(Rv64DivRemExecutor),
}

/// RISC-V 32-bit Io Instruction Executors
#[derive(Clone, Copy, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum Rv64IoExecutor {
    HintStore(Rv64HintStoreExecutor),
}

// ============ VmExtension Implementations ============

impl<F: PrimeField32> VmExecutionExtension<F> for Rv64I {
    type Executor = Rv64IExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Rv64IExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();

        let base_alu =
            Rv64BaseAluExecutor::new(Rv64BaseAluAdapterExecutor, BaseAluOpcode::CLASS_OFFSET);
        inventory.add_executor(base_alu, BaseAluOpcode::iter().map(|x| x.global_opcode()))?;

        let lt = LessThanExecutor::new(Rv64BaseAluAdapterExecutor, LessThanOpcode::CLASS_OFFSET);
        inventory.add_executor(lt, LessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let shift = ShiftExecutor::new(Rv64BaseAluAdapterExecutor, ShiftOpcode::CLASS_OFFSET);
        inventory.add_executor(shift, ShiftOpcode::iter().map(|x| x.global_opcode()))?;

        let load_store = LoadStoreExecutor::new(
            Rv64LoadStoreAdapterExecutor::new(pointer_max_bits),
            Rv64LoadStoreOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(
            load_store,
            Rv64LoadStoreOpcode::iter()
                .take(Rv64LoadStoreOpcode::STOREB as usize + 1)
                .map(|x| x.global_opcode()),
        )?;

        let load_sign_extend =
            LoadSignExtendExecutor::new(Rv64LoadStoreAdapterExecutor::new(pointer_max_bits));
        inventory.add_executor(
            load_sign_extend,
            [Rv64LoadStoreOpcode::LOADB, Rv64LoadStoreOpcode::LOADH].map(|x| x.global_opcode()),
        )?;

        let beq = BranchEqualExecutor::new(
            Rv64BranchAdapterExecutor,
            BranchEqualOpcode::CLASS_OFFSET,
            DEFAULT_PC_STEP,
        );
        inventory.add_executor(beq, BranchEqualOpcode::iter().map(|x| x.global_opcode()))?;

        let blt = BranchLessThanExecutor::new(
            Rv64BranchAdapterExecutor,
            BranchLessThanOpcode::CLASS_OFFSET,
        );
        inventory.add_executor(blt, BranchLessThanOpcode::iter().map(|x| x.global_opcode()))?;

        let jal_lui = Rv64JalLuiExecutor::new(Rv64CondRdWriteAdapterExecutor::new(
            Rv64RdWriteAdapterExecutor,
        ));
        inventory.add_executor(jal_lui, Rv64JalLuiOpcode::iter().map(|x| x.global_opcode()))?;

        let jalr = Rv64JalrExecutor::new(Rv64JalrAdapterExecutor);
        inventory.add_executor(jalr, Rv64JalrOpcode::iter().map(|x| x.global_opcode()))?;

        let auipc = Rv64AuipcExecutor::new(Rv64RdWriteAdapterExecutor);
        inventory.add_executor(auipc, Rv64AuipcOpcode::iter().map(|x| x.global_opcode()))?;

        // There is no downside to adding phantom sub-executors, so we do it in the base extension.
        inventory.add_phantom_sub_executor(
            phantom::Rv64HintInputSubEx,
            PhantomDiscriminant(Rv64Phantom::HintInput as u16),
        )?;
        inventory.add_phantom_sub_executor(
            phantom::Rv64HintRandomSubEx,
            PhantomDiscriminant(Rv64Phantom::HintRandom as u16),
        )?;
        inventory.add_phantom_sub_executor(
            phantom::Rv64PrintStrSubEx,
            PhantomDiscriminant(Rv64Phantom::PrintStr as u16),
        )?;
        inventory.add_phantom_sub_executor(
            phantom::Rv64HintLoadByKeySubEx,
            PhantomDiscriminant(Rv64Phantom::HintLoadByKey as u16),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Rv64I {
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

        let base_alu = Rv64BaseAluAir::new(
            Rv64BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            BaseAluCoreAir::new(bitwise_lu, BaseAluOpcode::CLASS_OFFSET),
        );
        inventory.add_air(base_alu);

        let lt = Rv64LessThanAir::new(
            Rv64BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            LessThanCoreAir::new(bitwise_lu, LessThanOpcode::CLASS_OFFSET),
        );
        inventory.add_air(lt);

        let shift = Rv64ShiftAir::new(
            Rv64BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu),
            ShiftCoreAir::new(bitwise_lu, range_checker, ShiftOpcode::CLASS_OFFSET),
        );
        inventory.add_air(shift);

        let load_store = Rv64LoadStoreAir::new(
            Rv64LoadStoreAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                pointer_max_bits,
            ),
            LoadStoreCoreAir::new(Rv64LoadStoreOpcode::CLASS_OFFSET),
        );
        inventory.add_air(load_store);

        let load_sign_extend = Rv64LoadSignExtendAir::new(
            Rv64LoadStoreAdapterAir::new(
                memory_bridge,
                exec_bridge,
                range_checker,
                pointer_max_bits,
            ),
            LoadSignExtendCoreAir::new(range_checker),
        );
        inventory.add_air(load_sign_extend);

        let beq = Rv64BranchEqualAir::new(
            Rv64BranchAdapterAir::new(exec_bridge, memory_bridge),
            BranchEqualCoreAir::new(BranchEqualOpcode::CLASS_OFFSET, DEFAULT_PC_STEP),
        );
        inventory.add_air(beq);

        let blt = Rv64BranchLessThanAir::new(
            Rv64BranchAdapterAir::new(exec_bridge, memory_bridge),
            BranchLessThanCoreAir::new(bitwise_lu, BranchLessThanOpcode::CLASS_OFFSET),
        );
        inventory.add_air(blt);

        let jal_lui = Rv64JalLuiAir::new(
            Rv64CondRdWriteAdapterAir::new(Rv64RdWriteAdapterAir::new(memory_bridge, exec_bridge)),
            Rv64JalLuiCoreAir::new(bitwise_lu),
        );
        inventory.add_air(jal_lui);

        let jalr = Rv64JalrAir::new(
            Rv64JalrAdapterAir::new(memory_bridge, exec_bridge),
            Rv64JalrCoreAir::new(bitwise_lu, range_checker),
        );
        inventory.add_air(jalr);

        let auipc = Rv64AuipcAir::new(
            Rv64RdWriteAdapterAir::new(memory_bridge, exec_bridge),
            Rv64AuipcCoreAir::new(bitwise_lu),
        );
        inventory.add_air(auipc);

        Ok(())
    }
}

pub struct Rv64ImCpuProverExt;
// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Rv64I> for Rv64ImCpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        _: &Rv64I,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let pointer_max_bits = inventory.airs().pointer_max_bits();
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
        inventory.next_air::<Rv64BaseAluAir>()?;
        let base_alu = Rv64BaseAluChip::new(
            BaseAluFiller::new(
                Rv64BaseAluAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                BaseAluOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(base_alu);

        inventory.next_air::<Rv64LessThanAir>()?;
        let lt = Rv64LessThanChip::new(
            LessThanFiller::new(
                Rv64BaseAluAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                LessThanOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(lt);

        inventory.next_air::<Rv64ShiftAir>()?;
        let shift = Rv64ShiftChip::new(
            ShiftFiller::new(
                Rv64BaseAluAdapterFiller::new(bitwise_lu.clone()),
                bitwise_lu.clone(),
                range_checker.clone(),
                ShiftOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(shift);

        inventory.next_air::<Rv64LoadStoreAir>()?;
        let load_store_chip = Rv64LoadStoreChip::new(
            LoadStoreFiller::new(
                Rv64LoadStoreAdapterFiller::new(pointer_max_bits, range_checker.clone()),
                Rv64LoadStoreOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_store_chip);

        inventory.next_air::<Rv64LoadSignExtendAir>()?;
        let load_sign_extend = Rv64LoadSignExtendChip::new(
            LoadSignExtendFiller::new(
                Rv64LoadStoreAdapterFiller::new(pointer_max_bits, range_checker.clone()),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(load_sign_extend);

        inventory.next_air::<Rv64BranchEqualAir>()?;
        let beq = Rv64BranchEqualChip::new(
            BranchEqualFiller::new(
                Rv64BranchAdapterFiller,
                BranchEqualOpcode::CLASS_OFFSET,
                DEFAULT_PC_STEP,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(beq);

        inventory.next_air::<Rv64BranchLessThanAir>()?;
        let blt = Rv64BranchLessThanChip::new(
            BranchLessThanFiller::new(
                Rv64BranchAdapterFiller,
                bitwise_lu.clone(),
                BranchLessThanOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(blt);

        inventory.next_air::<Rv64JalLuiAir>()?;
        let jal_lui = Rv64JalLuiChip::new(
            Rv64JalLuiFiller::new(
                Rv64CondRdWriteAdapterFiller::new(Rv64RdWriteAdapterFiller),
                bitwise_lu.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(jal_lui);

        inventory.next_air::<Rv64JalrAir>()?;
        let jalr = Rv64JalrChip::new(
            Rv64JalrFiller::new(
                Rv64JalrAdapterFiller,
                bitwise_lu.clone(),
                range_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(jalr);

        inventory.next_air::<Rv64AuipcAir>()?;
        let auipc = Rv64AuipcChip::new(
            Rv64AuipcFiller::new(Rv64RdWriteAdapterFiller, bitwise_lu.clone()),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(auipc);

        Ok(())
    }
}

impl<F> VmExecutionExtension<F> for Rv64M {
    type Executor = Rv64MExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Rv64MExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let mult =
            Rv64MultiplicationExecutor::new(Rv64MultAdapterExecutor, MulOpcode::CLASS_OFFSET);
        inventory.add_executor(mult, MulOpcode::iter().map(|x| x.global_opcode()))?;

        let mul_h = Rv64MulHExecutor::new(Rv64MultAdapterExecutor, MulHOpcode::CLASS_OFFSET);
        inventory.add_executor(mul_h, MulHOpcode::iter().map(|x| x.global_opcode()))?;

        let div_rem = Rv64DivRemExecutor::new(Rv64MultAdapterExecutor, DivRemOpcode::CLASS_OFFSET);
        inventory.add_executor(div_rem, DivRemOpcode::iter().map(|x| x.global_opcode()))?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Rv64M {
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

        let mult = Rv64MultiplicationAir::new(
            Rv64MultAdapterAir::new(exec_bridge, memory_bridge),
            MultiplicationCoreAir::new(range_tuple_checker, MulOpcode::CLASS_OFFSET),
        );
        inventory.add_air(mult);

        let mul_h = Rv64MulHAir::new(
            Rv64MultAdapterAir::new(exec_bridge, memory_bridge),
            MulHCoreAir::new(bitwise_lu, range_tuple_checker),
        );
        inventory.add_air(mul_h);

        let div_rem = Rv64DivRemAir::new(
            Rv64MultAdapterAir::new(exec_bridge, memory_bridge),
            DivRemCoreAir::new(bitwise_lu, range_tuple_checker, DivRemOpcode::CLASS_OFFSET),
        );
        inventory.add_air(div_rem);

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Rv64M> for Rv64ImCpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        extension: &Rv64M,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
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

        // These calls to next_air are not strictly necessary to construct the chips, but provide a
        // safeguard to ensure that chip construction matches the circuit definition
        inventory.next_air::<Rv64MultiplicationAir>()?;
        let mult = Rv64MultiplicationChip::new(
            MultiplicationFiller::new(
                Rv64MultAdapterFiller,
                range_tuple_checker.clone(),
                MulOpcode::CLASS_OFFSET,
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(mult);

        inventory.next_air::<Rv64MulHAir>()?;
        let mul_h = Rv64MulHChip::new(
            MulHFiller::new(
                Rv64MultAdapterFiller,
                bitwise_lu.clone(),
                range_tuple_checker.clone(),
            ),
            mem_helper.clone(),
        );
        inventory.add_executor_chip(mul_h);

        inventory.next_air::<Rv64DivRemAir>()?;
        let div_rem = Rv64DivRemChip::new(
            DivRemFiller::new(
                Rv64MultAdapterFiller,
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

impl<F> VmExecutionExtension<F> for Rv64Io {
    type Executor = Rv64IoExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Rv64IoExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let pointer_max_bits = inventory.pointer_max_bits();
        let hint_store =
            Rv64HintStoreExecutor::new(pointer_max_bits, Rv64HintStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            hint_store,
            Rv64HintStoreOpcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Rv64Io {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();

        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let pointer_max_bits = inventory.pointer_max_bits();

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

        let hint_store = Rv64HintStoreAir::new(
            exec_bridge,
            memory_bridge,
            bitwise_lu,
            Rv64HintStoreOpcode::CLASS_OFFSET,
            pointer_max_bits,
        );
        inventory.add_air(hint_store);

        Ok(())
    }
}

// This implementation is specific to CpuBackend because the lookup chips (VariableRangeChecker,
// BitwiseOperationLookupChip) are specific to CpuBackend.
impl<E, SC, RA> VmProverExtension<E, RA, Rv64Io> for Rv64ImCpuProverExt
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
{
    fn extend_prover(
        &self,
        _: &Rv64Io,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);
        let pointer_max_bits = inventory.airs().pointer_max_bits();

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

        inventory.next_air::<Rv64HintStoreAir>()?;
        let hint_store = Rv64HintStoreChip::new(
            Rv64HintStoreFiller::new(pointer_max_bits, bitwise_lu.clone()),
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

    pub struct Rv64HintInputSubEx;
    pub struct Rv64HintRandomSubEx;
    pub struct Rv64PrintStrSubEx;
    pub struct Rv64HintLoadByKeySubEx;

    impl<F: Field> PhantomSubExecutor<F> for Rv64HintInputSubEx {
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

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv64HintRandomSubEx {
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
            static WARN_ONCE: std::sync::Once = std::sync::Once::new();
            WARN_ONCE.call_once(|| {
                eprintln!("WARNING: Using fixed-seed RNG for deterministic randomness. Consider security implications for your use case.");
            });

            let len = read_rv32_register(memory, a) as usize;
            streams.hint_stream.clear();
            streams.hint_stream.extend(
                std::iter::repeat_with(|| F::from_canonical_u8(rng.gen::<u8>())).take(len * 4),
            );
            Ok(())
        }
    }

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv64PrintStrSubEx {
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

    impl<F: PrimeField32> PhantomSubExecutor<F> for Rv64HintLoadByKeySubEx {
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
                bail!("Rv64HintLoadByKey: key not found");
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
