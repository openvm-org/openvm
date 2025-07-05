use openvm_stark_backend::p3_field::PrimeField32;

use super::{ExecutionError, VmSegmentState};
use crate::arch::VmExecutionConfig;

/// Trait for execution control, determining segmentation and stopping conditions
/// Invariants:
/// - `ExecutionControl` should be stateless.
/// - For E3/E4, `ExecutionControl` is for a specific execution and cannot be used for another
///   execution with different inputs or segmentation criteria.
pub trait ExecutionControl<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
{
    /// Host context
    type Ctx;

    fn initialize_context(&self) -> Self::Ctx;

    /// Determines if execution should suspend
    fn should_suspend(&self, state: &mut VmSegmentState<F, Self::Ctx>) -> bool;

    /// Called before execution begins
    fn on_start(&self, state: &mut VmSegmentState<F, Self::Ctx>);

    /// Called after suspend or terminate
    fn on_suspend_or_terminate(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        exit_code: Option<u32>,
    );

    #[inline(always)]
    fn on_suspend(&self, state: &mut VmSegmentState<F, Self::Ctx>) {
        self.on_suspend_or_terminate(state, None);
    }

    #[inline(always)]
    fn on_terminate(&self, state: &mut VmSegmentState<F, Self::Ctx>, exit_code: u32) {
        self.on_suspend_or_terminate(state, Some(exit_code));
    }

    /// Execute the instruction at program address `pc`
    fn execute_instruction(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        pc: u32,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32;
}
