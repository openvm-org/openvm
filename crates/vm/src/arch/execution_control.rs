use super::{ExecutionError, VmSegmentState};
use crate::system::program::PcEntry;

/// Trait for execution control, determining segmentation and stopping conditions
/// Invariants:
/// - `ExecutionControl` should be stateless.
/// - For E3/E4, `ExecutionControl` is for a specific execution and cannot be used for another
///   execution with different inputs or segmentation criteria.
pub trait ExecutionControl<F, Executor> {
    /// Read/write random access memory
    type Memory;
    /// Host context
    type Ctx;

    /// Determines if execution should suspend
    fn should_suspend(&self, state: &mut VmSegmentState<F, Self::Memory, Self::Ctx>) -> bool;

    /// Called after suspend or terminate
    fn on_suspend_or_terminate(
        &self,
        state: &mut VmSegmentState<F, Self::Memory, Self::Ctx>,
        exit_code: Option<u32>,
    );

    #[inline(always)]
    fn on_suspend(&self, state: &mut VmSegmentState<F, Self::Memory, Self::Ctx>) {
        self.on_suspend_or_terminate(state, None);
    }

    #[inline(always)]
    fn on_terminate(&self, state: &mut VmSegmentState<F, Self::Memory, Self::Ctx>, exit_code: u32) {
        self.on_suspend_or_terminate(state, Some(exit_code));
    }

    /// Execute the instruction at program address `pc`
    fn execute_instruction(
        &self,
        state: &mut VmSegmentState<F, Self::Memory, Self::Ctx>,
        executor: &mut Executor,
        pc_entry: &PcEntry<F>,
    ) -> Result<(), ExecutionError>;
}
