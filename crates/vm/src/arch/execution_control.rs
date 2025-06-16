use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{ExecutionError, VmChipComplex, VmConfig, VmSegmentState};

/// Trait for execution control, determining segmentation and stopping conditions
/// Invariants:
/// - `ExecutionControl` should be stateless.
/// - For E3/E4, `ExecutionControl` is for a specific execution and cannot be used for another
///   execution with different inputs or segmentation criteria.
pub trait ExecutionControl<F, VC>
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    /// Host context
    type Ctx;

    fn initialize_context(&self) -> Self::Ctx;

    /// Determines if execution should suspend
    fn should_suspend(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool;

    /// Called before execution begins
    fn on_start(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    );

    /// Called after suspend or terminate
    fn on_suspend_or_terminate(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        exit_code: Option<u32>,
    );

    fn on_suspend(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        self.on_suspend_or_terminate(state, chip_complex, None);
    }

    fn on_terminate(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        exit_code: u32,
    ) {
        self.on_suspend_or_terminate(state, chip_complex, Some(exit_code));
    }

    /// Execute a single instruction
    // TODO(ayush): change instruction to Instruction<u32> / PInstruction
    fn execute_instruction(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32;
}
