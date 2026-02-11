use derive_more::derive::From;
use openvm_circuit::arch::{
    AirInventory, AirInventoryError, ExecutorInventoryBuilder, ExecutorInventoryError,
    VmCircuitExtension, VmExecutionExtension,
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor};
use openvm_instructions::LocalOpcode;
use openvm_rv64im_transpiler::{
    Rv64AuipcOpcode, Rv64BaseAluOpcode, Rv64BaseAluWOpcode, Rv64BranchEqualOpcode,
    Rv64BranchLessThanOpcode, Rv64DivRemOpcode, Rv64DivRemWOpcode, Rv64HintStoreOpcode,
    Rv64JalLuiOpcode, Rv64JalrOpcode, Rv64LessThanOpcode, Rv64LoadStoreOpcode, Rv64MulHOpcode,
    Rv64MulOpcode, Rv64MulWOpcode, Rv64ShiftOpcode, Rv64ShiftWOpcode,
};
use openvm_stark_backend::{config::StarkGenericConfig, p3_field::PrimeField32};
use serde::{Deserialize, Serialize};
use strum::IntoEnumIterator;

use crate::{
    Rv64AuipcExecutor, Rv64BaseAluExecutor, Rv64BaseAluWExecutor, Rv64BranchEqualExecutor,
    Rv64BranchLessThanExecutor, Rv64DivRemExecutor, Rv64DivRemWExecutor, Rv64HintStoreExecutor,
    Rv64JalLuiExecutor, Rv64JalrExecutor, Rv64LessThanExecutor, Rv64LoadSignExtendExecutor,
    Rv64LoadStoreExecutor, Rv64MulExecutor, Rv64MulHExecutor, Rv64MulWExecutor, Rv64ShiftExecutor,
    Rv64ShiftWExecutor,
};

/// RISC-V 64-bit Base Integer Extension (RV64I).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv64I;

/// RISC-V 64-bit Base (RV64I) Instruction Executors
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum Rv64IExecutor {
    BaseAlu(Rv64BaseAluExecutor),
    BaseAluW(Rv64BaseAluWExecutor),
    LessThan(Rv64LessThanExecutor),
    Shift(Rv64ShiftExecutor),
    ShiftW(Rv64ShiftWExecutor),
    BranchEqual(Rv64BranchEqualExecutor),
    BranchLessThan(Rv64BranchLessThanExecutor),
    JalLui(Rv64JalLuiExecutor),
    Jalr(Rv64JalrExecutor),
    Auipc(Rv64AuipcExecutor),
    LoadStore(Rv64LoadStoreExecutor),
    LoadSignExtend(Rv64LoadSignExtendExecutor),
}

impl<F: PrimeField32> VmExecutionExtension<F> for Rv64I {
    type Executor = Rv64IExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Rv64IExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let base_alu = Rv64BaseAluExecutor::new(Rv64BaseAluOpcode::CLASS_OFFSET);
        inventory.add_executor(
            base_alu,
            Rv64BaseAluOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let base_alu_w = Rv64BaseAluWExecutor::new(Rv64BaseAluWOpcode::CLASS_OFFSET);
        inventory.add_executor(
            base_alu_w,
            Rv64BaseAluWOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let less_than = Rv64LessThanExecutor::new(Rv64LessThanOpcode::CLASS_OFFSET);
        inventory.add_executor(
            less_than,
            Rv64LessThanOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let shift = Rv64ShiftExecutor::new(Rv64ShiftOpcode::CLASS_OFFSET);
        inventory.add_executor(shift, Rv64ShiftOpcode::iter().map(|x| x.global_opcode()))?;

        let shift_w = Rv64ShiftWExecutor::new(Rv64ShiftWOpcode::CLASS_OFFSET);
        inventory.add_executor(shift_w, Rv64ShiftWOpcode::iter().map(|x| x.global_opcode()))?;

        let branch_eq = Rv64BranchEqualExecutor::new(Rv64BranchEqualOpcode::CLASS_OFFSET);
        inventory.add_executor(
            branch_eq,
            Rv64BranchEqualOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let branch_lt = Rv64BranchLessThanExecutor::new(Rv64BranchLessThanOpcode::CLASS_OFFSET);
        inventory.add_executor(
            branch_lt,
            Rv64BranchLessThanOpcode::iter().map(|x| x.global_opcode()),
        )?;

        let jal_lui = Rv64JalLuiExecutor::new(Rv64JalLuiOpcode::CLASS_OFFSET);
        inventory.add_executor(jal_lui, Rv64JalLuiOpcode::iter().map(|x| x.global_opcode()))?;

        let jalr = Rv64JalrExecutor::new(Rv64JalrOpcode::CLASS_OFFSET);
        inventory.add_executor(jalr, Rv64JalrOpcode::iter().map(|x| x.global_opcode()))?;

        let auipc = Rv64AuipcExecutor::new(Rv64AuipcOpcode::CLASS_OFFSET);
        inventory.add_executor(auipc, Rv64AuipcOpcode::iter().map(|x| x.global_opcode()))?;

        let loadstore = Rv64LoadStoreExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            loadstore,
            Rv64LoadStoreOpcode::iter()
                .take(Rv64LoadStoreOpcode::STOREB as usize + 1)
                .map(|x| x.global_opcode()),
        )?;

        let load_sign_extend = Rv64LoadSignExtendExecutor::new(Rv64LoadStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            load_sign_extend,
            [
                Rv64LoadStoreOpcode::LOADB,
                Rv64LoadStoreOpcode::LOADH,
                Rv64LoadStoreOpcode::LOADW,
            ]
            .into_iter()
            .map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Rv64I {
    fn extend_circuit(&self, _inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        Ok(())
    }
}

/// RISC-V 64-bit Multiplication Extension (RV64M).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv64M;

/// RISC-V Extension for handling IO (not to be confused with I base extension)
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Rv64Io;

/// RISC-V 64-bit M-extension Instruction Executors
#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum Rv64MExecutor {
    Mul(Rv64MulExecutor),
    MulH(Rv64MulHExecutor),
    DivRem(Rv64DivRemExecutor),
    MulW(Rv64MulWExecutor),
    DivRemW(Rv64DivRemWExecutor),
}

/// RISC-V 64-bit Io Instruction Executors
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

impl<F: PrimeField32> VmExecutionExtension<F> for Rv64M {
    type Executor = Rv64MExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Rv64MExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let mul = Rv64MulExecutor::new(Rv64MulOpcode::CLASS_OFFSET);
        inventory.add_executor(mul, Rv64MulOpcode::iter().map(|x| x.global_opcode()))?;

        let mulh = Rv64MulHExecutor::new(Rv64MulHOpcode::CLASS_OFFSET);
        inventory.add_executor(mulh, Rv64MulHOpcode::iter().map(|x| x.global_opcode()))?;

        let divrem = Rv64DivRemExecutor::new(Rv64DivRemOpcode::CLASS_OFFSET);
        inventory.add_executor(divrem, Rv64DivRemOpcode::iter().map(|x| x.global_opcode()))?;

        let mul_w = Rv64MulWExecutor::new(Rv64MulWOpcode::CLASS_OFFSET);
        inventory.add_executor(mul_w, Rv64MulWOpcode::iter().map(|x| x.global_opcode()))?;

        let divrem_w = Rv64DivRemWExecutor::new(Rv64DivRemWOpcode::CLASS_OFFSET);
        inventory.add_executor(
            divrem_w,
            Rv64DivRemWOpcode::iter().map(|x| x.global_opcode()),
        )?;

        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Rv64M {
    fn extend_circuit(&self, _inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        Ok(())
    }
}

impl<F: PrimeField32> VmExecutionExtension<F> for Rv64Io {
    type Executor = Rv64IoExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Rv64IoExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        let hint_store = Rv64HintStoreExecutor::new(Rv64HintStoreOpcode::CLASS_OFFSET);
        inventory.add_executor(
            hint_store,
            Rv64HintStoreOpcode::iter().map(|x| x.global_opcode()),
        )?;
        Ok(())
    }
}

impl<SC: StarkGenericConfig> VmCircuitExtension<SC> for Rv64Io {
    fn extend_circuit(&self, _inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        Ok(())
    }
}
